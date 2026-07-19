//! Cross-component dynamic-linker call machinery.
//!
//! When one component in a workload imports a function that another component
//! exports, the linker wires the import to one of the `invoke_*` helpers here.
//! Each call is dispatched, by signature, down one of two paths:
//!
//! - the **shared-store path** ([`invoke_shared_store_linked_export`] /
//!   [`invoke_linked_sync_export`]), where the callee was pre-instantiated into
//!   the caller's long-lived store and handles can cross the boundary by
//!   identity, and
//! - the **ephemeral path** ([`invoke_ephemeral_linked_export`]), where a
//!   plain-value call runs in a throwaway store built per call.
//!
//! Store creation for both paths is also here: [`ComponentCtxTemplate`] is the
//! cheap recipe for a component's [`Ctx`], [`build_ctx_from_template`] turns one
//! into a [`Ctx`], and [`new_store_from_templates`] / [`new_ephemeral_store`]
//! assemble the store (pre-instantiating the linked components). See
//! [`EphemeralLinkedCall`] for how the ephemeral path is captured at link time.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use tokio::sync::RwLock;
use tokio::time::timeout;
use tracing::trace;
use wasmtime::component::{
    Accessor, ComponentExportIndex, InstancePre, Val,
    types::{ComponentFunc, Type},
};
use wasmtime::error::Context as _;
use wasmtime::{AsContext, AsContextMut, StoreContextMut};
use wasmtime_wasi::WasiCtxBuilder;

#[cfg(feature = "wasi-tls")]
use crate::engine::ctx::SharedTlsProvider;
use crate::engine::ctx::{
    AccessorActiveCtxGuard, Ctx, SharedCtx, StoreActiveCtxGuard, WamnStoreLimiter,
};
use crate::engine::value::{carries_cross_store_handle, lift_results, lower_params};
use crate::engine::volumes::{ResolvedVolumeMount, resolve_component_volume_mounts_in_map};
use crate::engine::workload::{WorkloadComponent, WorkloadMetadata};
use crate::plugin::HostPlugin;
use crate::sockets::{self, SocketAddrUse, loopback};

/// A cheap, cloneable recipe for building a component's [`Ctx`].
///
/// Constructing a [`Ctx`] is comparatively expensive (it canonicalizes volume
/// mounts, builds a fresh `WasiCtx`, sockets ctx, etc.), and a single store may
/// need a ctx for the active component *and* for each component linked into it.
/// Rather than re-derive those inputs from [`WorkloadMetadata`] every time, we
/// snapshot the per-component pieces once into this template via
/// [`ComponentCtxTemplate::from_metadata`] and hand it to
/// [`build_ctx_from_template`], which turns it into an actual [`Ctx`] for a
/// given `store_id`.
///
/// Templates drive store creation on both linked-call paths:
/// [`new_store_from_templates`] builds the long-lived request/service store
/// (one active template + the linked templates), and the ephemeral path
/// rebuilds templates per call from metadata inside [`new_ephemeral_store`].
/// The `tls_provider` field is populated (under `wasi-tls`) at the
/// [`EphemeralLinkedCall`] construction site so the ephemeral path doesn't drop
/// TLS support that the request path has.
#[derive(Clone)]
pub(crate) struct ComponentCtxTemplate {
    component_id: Arc<str>,
    workload_id: Arc<str>,
    local_resources: crate::types::LocalResources,
    volume_mounts: Vec<ResolvedVolumeMount>,
    plugins: Option<HashMap<&'static str, Arc<dyn HostPlugin + Send + Sync>>>,
    loopback: Arc<std::sync::Mutex<loopback::Network>>,
    #[cfg(feature = "wasi-tls")]
    tls_provider: Option<SharedTlsProvider>,
}

impl ComponentCtxTemplate {
    fn from_metadata(metadata: &WorkloadMetadata) -> Self {
        Self {
            component_id: metadata.id.clone(),
            workload_id: metadata.workload_id.clone(),
            local_resources: metadata.local_resources.clone(),
            volume_mounts: metadata.resolved_volume_mounts.clone(),
            plugins: metadata.plugins.clone(),
            loopback: metadata.loopback.clone(),
            #[cfg(feature = "wasi-tls")]
            tls_provider: None,
        }
    }
}

#[cfg(not(feature = "wasi-tls"))]
pub(crate) fn component_ctx_template_from_metadata(
    metadata: &WorkloadMetadata,
) -> ComponentCtxTemplate {
    ComponentCtxTemplate::from_metadata(metadata)
}

#[cfg(feature = "wasi-tls")]
pub(crate) fn component_ctx_template_from_metadata_with_tls(
    metadata: &WorkloadMetadata,
    tls_provider: Option<SharedTlsProvider>,
) -> ComponentCtxTemplate {
    let mut template = ComponentCtxTemplate::from_metadata(metadata);
    template.tls_provider = tls_provider;
    template
}

/// Everything needed to spin up a throwaway store for a single cross-component
/// linked call.
///
/// # Where it fits in a cross-component call
///
/// When a component (`active_component_id`) imports a function that another
/// component in the same workload exports, the dynamic linker routes the call
/// to one of two paths, chosen at link time by [`func_is_ephemeral_safe`]:
///
/// - **Shared-store path** — used when the call's signature carries a handle
///   that must keep its identity across the boundary (resource/borrow/stream/
///   future/error-context; see [`carries_cross_store_handle`]). The callee is
///   instantiated once into the caller's long-lived store and reused
///   ([`invoke_shared_store_linked_export`]).
/// - **Ephemeral path** — used when every parameter and result is a *plain
///   value* (no cross-store handle). The call runs in a brand-new store that is
///   instantiated, invoked, and dropped per call
///   ([`invoke_ephemeral_linked_export`]), so its core-instance slots are
///   reclaimed immediately. Plain values copy cleanly across the store
///   boundary, so nothing is lost by not sharing a store.
///
/// This struct is the captured input for that second path. One
/// `Arc<EphemeralLinkedCall>` is built per eligible import during
/// `link_components` and stored on the [`LinkedExportInvocation`]; each call
/// hands it to [`new_ephemeral_store`], which rebuilds the active + linked
/// [`ComponentCtxTemplate`]s from current metadata (`components`),
/// pre-instantiates the linked components into the fresh store, and runs the
/// export. Wrapped in `Arc` so the per-call clone is a pointer bump rather than
/// a deep copy of the engine/handler/component map.
#[derive(Clone)]
pub(crate) struct EphemeralLinkedCall {
    pub(crate) engine: wasmtime::Engine,
    pub(crate) http_handler: Arc<dyn crate::host::http::HostHandler>,
    pub(crate) components: Arc<RwLock<HashMap<Arc<str>, WorkloadComponent>>>,
    pub(crate) active_component_id: Arc<str>,
    pub(crate) linked_component_ids: Vec<Arc<str>>,
    #[cfg(feature = "wasi-tls")]
    pub(crate) tls_provider: Option<SharedTlsProvider>,
}

fn type_is_ephemeral_safe(ty: &Type) -> bool {
    !carries_cross_store_handle(ty)
}

pub(crate) fn func_is_ephemeral_safe(func_ty: &ComponentFunc) -> bool {
    func_ty.params().all(|(_, ty)| type_is_ephemeral_safe(&ty))
        && func_ty.results().all(|ty| type_is_ephemeral_safe(&ty))
}

/// wamn carried patch: resolve the raw-sockets opt-in for a component.
///
/// The component's `wamn.allow-raw-sockets` config wins, then the
/// `WAMN_ALLOW_RAW_SOCKETS` env var, else DENY. An unparseable value denies
/// (this is a security floor). Pulled out of `build_ctx_from_template` so the
/// precedence and parse-fail-closed behavior are unit-testable without touching
/// process env. The precedent is the carried epoch/memory-limiter config reads.
fn resolve_allow_raw_sockets(config: Option<&str>, env: Option<&str>) -> bool {
    config
        .map(|v| v.parse::<bool>().unwrap_or(false))
        .or_else(|| env.map(|v| v.parse::<bool>().unwrap_or(false)))
        .unwrap_or(false)
}

/// wamn carried patch: the `socket_addr_check` policy, as a pure decision.
///
/// `wasi:sockets` is linked into every component unconditionally (see
/// `engine/mod.rs`) and the parsed egress allowlist (`allowed_hosts`) governs
/// the `wasi:http` path only, so a guest could otherwise open a raw socket to
/// any post-DNS address and bypass egress policy. This is the platform-plan 8.2
/// deny-all posture at this layer:
///
/// - **Bind** (`TcpBind`/`UdpBind`): only a service component may bind, and only
///   to loopback; every non-service bind and every non-loopback bind is denied.
/// - **Raw egress** (`TcpConnect`/`UdpConnect`/`UdpOutgoingDatagram`): denied
///   unless the workload opts in via `allow_raw_sockets`. These reach the check
///   with a post-DNS `SocketAddr`, not a name, so allowlist matching cannot be
///   applied here without hooking `ip_name_lookup` (name->IP allowlists are
///   fragile); a binary deny-unless-opt-in is the honest policy at this layer.
fn socket_addr_permitted(
    reason: SocketAddrUse,
    ip_is_loopback: bool,
    is_service: bool,
    allow_raw_sockets: bool,
) -> bool {
    match reason {
        SocketAddrUse::TcpBind | SocketAddrUse::UdpBind => is_service && ip_is_loopback,
        SocketAddrUse::TcpConnect
        | SocketAddrUse::UdpConnect
        | SocketAddrUse::UdpOutgoingDatagram => allow_raw_sockets,
    }
}

/// True for the raw outbound-egress socket uses gated by `allow_raw_sockets`.
fn is_raw_egress(reason: SocketAddrUse) -> bool {
    matches!(
        reason,
        SocketAddrUse::TcpConnect | SocketAddrUse::UdpConnect | SocketAddrUse::UdpOutgoingDatagram
    )
}

async fn build_ctx_from_template(
    template: &ComponentCtxTemplate,
    http_handler: Arc<dyn crate::host::http::HostHandler>,
    all_volume_mounts: &[ResolvedVolumeMount],
    store_id: &str,
    is_service: bool,
) -> anyhow::Result<Ctx> {
    let mut wasi_ctx_builder = WasiCtxBuilder::new();
    wasi_ctx_builder
        .envs(
            template
                .local_resources
                .environment
                .iter()
                .map(|kv| (kv.0.as_str(), kv.1.as_str()))
                .collect::<Vec<_>>()
                .as_slice(),
        )
        .inherit_stdout()
        .inherit_stderr();

    // wamn carried patch: `wasi:sockets` is linked into every component
    // unconditionally (see `engine/mod.rs`), and the parsed egress allowlist
    // (`allowed_hosts`) governs the `wasi:http` path only. So a guest could open
    // a raw socket to any post-DNS address and bypass egress policy entirely.
    // The policy is factored into `socket_addr_permitted` (bind is
    // service-loopback-only; raw egress -- `TcpConnect`/`UdpConnect`/
    // `UdpOutgoingDatagram` -- is denied unless the workload opts in) and the
    // opt-in into `resolve_allow_raw_sockets` (config > env > DENY, unparseable
    // denies -- a security floor). The default-deny is visible: on the first raw
    // egress denial we `warn!` once per component so an operator can diagnose a
    // blocked node. Follow-up to the `TcpConnect`-only opt-in: `UdpConnect`/
    // `UdpOutgoingDatagram` now share the same gate, and `UdpBind` is tightened
    // from loopback-or-unspecified-for-any-component to match `TcpBind`.
    let allow_raw_sockets = resolve_allow_raw_sockets(
        template
            .local_resources
            .config
            .get("wamn.allow-raw-sockets")
            .map(String::as_str),
        std::env::var("WAMN_ALLOW_RAW_SOCKETS").ok().as_deref(),
    );
    let raw_socket_component = template.component_id.clone();
    let raw_socket_denial_logged = Arc::new(AtomicBool::new(false));

    let sockets_ctx = sockets::WasiSocketsCtx {
        socket_addr_check: sockets::SocketAddrCheck::new(move |addr, reason| {
            let raw_socket_component = raw_socket_component.clone();
            let raw_socket_denial_logged = raw_socket_denial_logged.clone();
            Box::pin(async move {
                let permitted = socket_addr_permitted(
                    reason,
                    addr.ip().is_loopback(),
                    is_service,
                    allow_raw_sockets,
                );
                if !permitted
                    && is_raw_egress(reason)
                    && !raw_socket_denial_logged.swap(true, Ordering::Relaxed)
                {
                    tracing::warn!(
                        target: "wamn::sockets",
                        component = %raw_socket_component,
                        addr = %addr,
                        reason = ?reason,
                        "wasi:sockets raw egress denied: workload has not opted into \
                         raw sockets (set wamn.allow-raw-sockets=true or \
                         WAMN_ALLOW_RAW_SOCKETS=true); egress allowlists govern \
                         wasi:http only"
                    );
                }
                permitted
            })
        }),
        loopback: Arc::clone(&template.loopback),
        ..Default::default()
    };

    for mount in all_volume_mounts {
        wasi_ctx_builder.preopened_dir(
            &mount.host_path,
            &mount.mount_path,
            mount.dir_perms,
            mount.file_perms,
        )?;
    }

    let mut ctx_builder = Ctx::builder(template.workload_id.clone(), template.component_id.clone())
        .with_http_handler(http_handler)
        .with_wasi_ctx(wasi_ctx_builder.build())
        .with_sockets(sockets_ctx)
        .with_allowed_hosts(template.local_resources.allowed_hosts.clone());

    if let Some(plugins) = &template.plugins {
        ctx_builder = ctx_builder.with_plugins(plugins.clone());
    }

    #[cfg(feature = "wasi-tls")]
    if let Some(provider) = template.tls_provider.clone() {
        ctx_builder = ctx_builder.with_tls_provider(provider);
    }

    let mut ctx = ctx_builder.build();
    ctx.store_id = store_id.to_string().into();
    Ok(ctx)
}

pub(crate) async fn new_store_from_templates(
    engine: &wasmtime::Engine,
    http_handler: Arc<dyn crate::host::http::HostHandler>,
    active: &ComponentCtxTemplate,
    linked: &[ComponentCtxTemplate],
    linked_instances: &[(Arc<str>, InstancePre<SharedCtx>)],
    is_service: bool,
) -> anyhow::Result<wasmtime::Store<SharedCtx>> {
    let store_id = uuid::Uuid::new_v4().to_string();
    let all_volume_mounts = std::iter::once(active)
        .chain(linked.iter())
        .flat_map(|template| template.volume_mounts.clone())
        .collect::<Vec<_>>();
    let active_ctx = build_ctx_from_template(
        active,
        http_handler.clone(),
        &all_volume_mounts,
        &store_id,
        is_service,
    )
    .await?;
    let mut shared_ctx = SharedCtx::new(active_ctx);

    for linked in linked {
        let linked_ctx = build_ctx_from_template(
            linked,
            http_handler.clone(),
            &all_volume_mounts,
            &store_id,
            false,
        )
        .await?;
        shared_ctx
            .contexts
            .insert(linked.component_id.clone(), linked_ctx);
    }

    let mut store = wasmtime::Store::new(engine, shared_ctx);

    // wamn carried patch: stores default to an epoch deadline of 0, so a host
    // that enables `Config::epoch_interruption` and drives
    // `Engine::increment_epoch` would trap every guest on the first tick.
    // Give each store a deadline in engine epoch ticks: the active
    // component's `wamn.epoch-deadline-ticks` config wins, then the
    // WAMN_EPOCH_DEADLINE_TICKS env var, then effectively-unbounded
    // (u64::MAX would wrap in wasmtime's `current_epoch + delta`).
    let epoch_deadline_ticks = active
        .local_resources
        .config
        .get("wamn.epoch-deadline-ticks")
        .and_then(|v| v.parse::<u64>().ok())
        .or_else(|| {
            std::env::var("WAMN_EPOCH_DEADLINE_TICKS")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
        })
        .unwrap_or(u64::MAX / 2);
    store.set_epoch_deadline(epoch_deadline_ticks);

    // wamn carried patch: per-component linear-memory budget, enforced below
    // the pooling allocator's engine-wide ceiling via `Store::limiter` (see
    // `WamnStoreLimiter`). Resolution order: the workload spec's first-class
    // `memory_limit_mb` (<= 0 means unset upstream), then the active
    // component's `wamn.memory-limit-mb` config, then the
    // WAMN_MEMORY_LIMIT_MB env var. With no budget configured no limiter is
    // attached — unbudgeted stores are byte-identical to upstream.
    let memory_budget_mb = (active.local_resources.memory_limit_mb > 0)
        .then_some(active.local_resources.memory_limit_mb as u64)
        .or_else(|| {
            active
                .local_resources
                .config
                .get("wamn.memory-limit-mb")
                .and_then(|v| v.parse::<u64>().ok())
        })
        .or_else(|| {
            std::env::var("WAMN_MEMORY_LIMIT_MB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
        });
    if let Some(mb) = memory_budget_mb {
        // A budget above the host ceiling is a hard configuration error,
        // never a silent clamp. The ceiling is advertised by the embedding
        // host via WAMN_MEMORY_CEILING_MB (the pooling allocator's
        // max_memory_size is not introspectable from the engine); hosts that
        // do not advertise one skip the check.
        if let Some(ceiling_mb) = std::env::var("WAMN_MEMORY_CEILING_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
        {
            if mb > ceiling_mb {
                anyhow::bail!(
                    "component '{}': wamn memory budget {mb} MiB exceeds the host ceiling \
                     {ceiling_mb} MiB (pooling max_memory_size); lower the budget or raise \
                     the ceiling",
                    active.component_id,
                );
            }
        }
        store.data_mut().wamn_limiter =
            WamnStoreLimiter::new((mb as usize) << 20, active.component_id.clone());
        store.limiter(|ctx| &mut ctx.wamn_limiter);
    }

    let active_id = active.component_id.clone();
    for (linked_id, linked_pre) in linked_instances {
        store.data_mut().set_active_ctx(linked_id)?;
        let instantiate_result = linked_pre.instantiate_async(&mut store).await;
        store.data_mut().set_active_ctx(&active_id)?;
        let instance = instantiate_result.map_err(|e| {
            anyhow::anyhow!(
                "failed to instantiate linked component '{linked_id}' in ephemeral store: {e}"
            )
        })?;
        store
            .data_mut()
            .exporter_instances
            .insert(linked_id.clone(), instance);
    }

    Ok(store)
}

async fn new_ephemeral_store(
    call: &EphemeralLinkedCall,
) -> anyhow::Result<wasmtime::Store<SharedCtx>> {
    let mut component_ids = call.linked_component_ids.clone();
    component_ids.push(call.active_component_id.clone());
    component_ids.sort();
    component_ids.dedup();
    resolve_component_volume_mounts_in_map(&call.components, &component_ids).await?;

    let (active_metadata, linked_metadata) = {
        let components = call.components.read().await;
        let active = components
            .get(&call.active_component_id)
            .with_context(|| {
                format!(
                    "ephemeral linked component '{}' not found",
                    call.active_component_id
                )
            })?
            .metadata
            .clone();
        let linked = call
            .linked_component_ids
            .iter()
            .map(|component_id| {
                components
                    .get(component_id)
                    .with_context(|| format!("linked component '{component_id}' not found"))
                    .map(|component| component.metadata.clone())
            })
            .collect::<wasmtime::Result<Vec<_>>>()?;
        (active, linked)
    };

    #[cfg(not(feature = "wasi-tls"))]
    let active = component_ctx_template_from_metadata(&active_metadata);
    #[cfg(feature = "wasi-tls")]
    let active =
        component_ctx_template_from_metadata_with_tls(&active_metadata, call.tls_provider.clone());

    #[cfg(not(feature = "wasi-tls"))]
    let linked = linked_metadata
        .iter()
        .map(component_ctx_template_from_metadata)
        .collect::<Vec<_>>();
    #[cfg(feature = "wasi-tls")]
    let linked = linked_metadata
        .iter()
        .map(|metadata| {
            component_ctx_template_from_metadata_with_tls(metadata, call.tls_provider.clone())
        })
        .collect::<Vec<_>>();

    let linked_instances = {
        let mut components = call.components.write().await;
        call.linked_component_ids
            .iter()
            .map(|component_id| {
                let component = components.get_mut(component_id).ok_or_else(|| {
                    wasmtime::format_err!("linked component '{component_id}' not found")
                })?;
                component
                    .pre_instantiate()
                    .map(|pre| (component_id.clone(), pre))
            })
            .collect::<wasmtime::Result<Vec<_>>>()
            .map_err(|e| {
                anyhow::anyhow!(
                    "failed to pre-instantiate linked components for ephemeral call: {e}"
                )
            })?
    };

    new_store_from_templates(
        &call.engine,
        call.http_handler.clone(),
        &active,
        &linked,
        &linked_instances,
        false,
    )
    .await
}

#[derive(Clone)]
pub(crate) struct LinkedExportInvocation {
    pub(crate) import_name: Arc<str>,
    pub(crate) export_name: Arc<str>,
    pub(crate) pre: InstancePre<SharedCtx>,
    pub(crate) plugin_component_id: Arc<str>,
    pub(crate) func_idx: ComponentExportIndex,
    pub(crate) param_tys: Arc<std::sync::OnceLock<Arc<[Type]>>>,
    pub(crate) ephemeral_call: Option<Arc<EphemeralLinkedCall>>,
}

pub(crate) async fn invoke_linked_async_export(
    accessor: &Accessor<SharedCtx>,
    params: &[Val],
    results: &mut [Val],
    inv: &LinkedExportInvocation,
) -> wasmtime::Result<()> {
    if let Some(ephemeral_call) = &inv.ephemeral_call {
        invoke_ephemeral_linked_export(params, results, inv, ephemeral_call).await
    } else {
        invoke_shared_store_linked_export(accessor, params, results, inv).await
    }
}

/// Aborts the wrapped task when dropped before it completes, so a cancelled
/// caller (e.g. a client disconnect tearing down the request future) reclaims
/// the ephemeral store's core-instance slots immediately instead of leaving a
/// detached task to run to its timeout.
struct AbortOnDrop<T>(tokio::task::JoinHandle<T>);

impl<T> Drop for AbortOnDrop<T> {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Run a plain-value async linked call in a short-lived store that is dropped
/// (reclaiming its core-instance slots) as soon as the call returns.
async fn invoke_ephemeral_linked_export(
    params: &[Val],
    results: &mut [Val],
    inv: &LinkedExportInvocation,
    ephemeral_call: &EphemeralLinkedCall,
) -> wasmtime::Result<()> {
    let mut store = new_ephemeral_store(ephemeral_call)
        .await
        .map_err(|e| wasmtime::format_err!("{e:#}"))?;

    let params_buf = params.to_vec();
    let mut results_buf = vec![Val::Bool(false); results.len()];
    let call_import_name = inv.import_name.clone();
    let call_export_name = inv.export_name.clone();
    let call_pre = inv.pre.clone();
    let func_idx = inv.func_idx;

    trace!(
        name = %inv.import_name,
        fn_name = %inv.export_name,
        ?params,
        "invoking ephemeral dynamic export"
    );

    let mut task = AbortOnDrop(tokio::task::spawn(async move {
        let instance = call_pre.instantiate_async(&mut store).await?;
        store
            .run_concurrent(async move |accessor| {
                let func = accessor.with(|mut access| -> wasmtime::Result<_> {
                    instance.get_func(&mut access, func_idx).with_context(|| {
                        format!(
                            "function not found for linked import {call_import_name}.{call_export_name}"
                        )
                    })
                })?;
                const CALL_TIMEOUT: Duration = Duration::from_secs(600);
                timeout(
                    CALL_TIMEOUT,
                    func.call_concurrent(accessor, &params_buf, &mut results_buf),
                )
                .await
                .map_err(|e| {
                    wasmtime::format_err!("function call timed out after 600 seconds: {e}")
                })??;
                Ok::<Vec<Val>, wasmtime::Error>(results_buf)
            })
            .await
            .map_err(|e| wasmtime::format_err!("{e:#}"))?
    }));
    let call_result = (&mut task.0)
        .await
        .map_err(|e| wasmtime::format_err!("ephemeral linked call task failed: {e}"));
    let call_result = call_result??;

    for (i, v) in call_result.into_iter().enumerate() {
        *results.get_mut(i).context("result index out of bounds")? = v;
    }

    trace!(
        name = %inv.import_name,
        fn_name = %inv.export_name,
        ?results,
        "invoked ephemeral dynamic export"
    );

    Ok(())
}

async fn invoke_shared_store_linked_export(
    accessor: &Accessor<SharedCtx>,
    params: &[Val],
    results: &mut [Val],
    inv: &LinkedExportInvocation,
) -> wasmtime::Result<()> {
    let _active_ctx = AccessorActiveCtxGuard::new(accessor, &inv.plugin_component_id)?;

    let call: wasmtime::Result<()> = async {
        let (func, params_buf) = accessor.with(|mut access| -> wasmtime::Result<_> {
            let instance = access
                .data_mut()
                .exporter_instances
                .get(&inv.plugin_component_id)
                .copied()
                .with_context(|| {
                    format!(
                        "linked component '{}' was not pre-instantiated in this store",
                        inv.plugin_component_id
                    )
                })?;
            let func = instance
                .get_func(&mut access, inv.func_idx)
                .context("function not found")?;
            let tys = inv.param_tys.get_or_init(|| {
                func.ty(access.as_context())
                    .params()
                    .map(|(_, ty)| ty)
                    .collect::<Vec<_>>()
                    .into()
            });
            let params_buf = lower_params(&mut access.as_context_mut(), params, tys)?;
            Ok((func, params_buf))
        })?;

        trace!(name = %inv.import_name, fn_name = %inv.export_name, "invoking dynamic export");

        let mut results_buf = vec![Val::Bool(false); results.len()];
        func.call_concurrent(accessor, &params_buf, &mut results_buf)
            .await?;

        accessor.with(|mut access| -> wasmtime::Result<_> {
            lift_results(&mut access.as_context_mut(), results_buf, results)
        })?;

        Ok(())
    }
    .await;

    call?;

    trace!(name = %inv.import_name, fn_name = %inv.export_name, "invoked dynamic export");

    Ok(())
}

pub(crate) async fn invoke_linked_sync_export(
    store: StoreContextMut<'_, SharedCtx>,
    params: &[Val],
    results: &mut [Val],
    inv: &LinkedExportInvocation,
) -> wasmtime::Result<()> {
    let mut active_ctx = StoreActiveCtxGuard::new(store, &inv.plugin_component_id)?;
    let mut store = active_ctx.store_mut();

    async {
        let instance = store
            .data()
            .exporter_instances
            .get(&inv.plugin_component_id)
            .copied()
            .with_context(|| {
                format!(
                    "linked component '{}' was not pre-instantiated in this store",
                    inv.plugin_component_id
                )
            })?;

        let func = instance
            .get_func(&mut store, inv.func_idx)
            .context("function not found")?;
        let tys = inv.param_tys.get_or_init(|| {
            func.ty(store.as_context())
                .params()
                .map(|(_, ty)| ty)
                .collect::<Vec<_>>()
                .into()
        });
        let params_buf = lower_params(store, params, tys)?;
        trace!(name = %inv.import_name, fn_name = %inv.export_name, "invoking dynamic export");

        let mut results_buf = vec![Val::Bool(false); results.len()];

        const CALL_TIMEOUT: Duration = Duration::from_secs(30);
        timeout(
            CALL_TIMEOUT,
            func.call_async(&mut store, &params_buf, &mut results_buf),
        )
        .await
        .map_err(|e| wasmtime::format_err!("function call timed out after 30 seconds: {e}"))??;

        lift_results(store, results_buf, results)?;
        trace!(name = %inv.import_name, fn_name = %inv.export_name, "invoked dynamic export");
        Ok(())
    }
    .await
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    // --- resolve_allow_raw_sockets: config > env > DENY; unparseable denies ---

    #[test]
    fn raw_sockets_denied_by_default() {
        assert!(!resolve_allow_raw_sockets(None, None));
    }

    #[test]
    fn raw_sockets_config_true_allows() {
        assert!(resolve_allow_raw_sockets(Some("true"), None));
    }

    #[test]
    fn raw_sockets_config_false_denies() {
        assert!(!resolve_allow_raw_sockets(Some("false"), None));
    }

    #[test]
    fn raw_sockets_env_true_allows_when_config_absent() {
        assert!(resolve_allow_raw_sockets(None, Some("true")));
    }

    #[test]
    fn raw_sockets_config_wins_over_env() {
        // A present config value short-circuits the env fallback: config `false`
        // denies even when the env var says `true`.
        assert!(!resolve_allow_raw_sockets(Some("false"), Some("true")));
    }

    #[test]
    fn raw_sockets_unparseable_config_denies() {
        // Security floor: an unparseable config value denies (and, being present,
        // does not fall through to the env var).
        assert!(!resolve_allow_raw_sockets(Some("yes"), Some("true")));
    }

    #[test]
    fn raw_sockets_unparseable_env_denies() {
        assert!(!resolve_allow_raw_sockets(None, Some("1")));
    }

    // --- socket_addr_permitted: bind posture ---

    #[test]
    fn tcp_bind_service_loopback_allowed() {
        assert!(socket_addr_permitted(
            SocketAddrUse::TcpBind,
            true,
            true,
            false
        ));
    }

    #[test]
    fn tcp_bind_service_non_loopback_denied() {
        assert!(!socket_addr_permitted(
            SocketAddrUse::TcpBind,
            false,
            true,
            false
        ));
    }

    #[test]
    fn tcp_bind_non_service_denied() {
        assert!(!socket_addr_permitted(
            SocketAddrUse::TcpBind,
            true,
            false,
            false
        ));
    }

    #[test]
    fn udp_bind_service_loopback_allowed() {
        // UdpBind now mirrors TcpBind (was loopback-or-unspecified for every
        // component -- the E16 asymmetry).
        assert!(socket_addr_permitted(
            SocketAddrUse::UdpBind,
            true,
            true,
            false
        ));
    }

    #[test]
    fn udp_bind_service_non_loopback_denied() {
        assert!(!socket_addr_permitted(
            SocketAddrUse::UdpBind,
            false,
            true,
            false
        ));
    }

    #[test]
    fn udp_bind_non_service_denied() {
        // Previously a non-service component could bind 0.0.0.0/loopback UDP; now
        // denied, matching TcpBind's non-service arm.
        assert!(!socket_addr_permitted(
            SocketAddrUse::UdpBind,
            true,
            false,
            false
        ));
    }

    // --- socket_addr_permitted: raw egress opt-in (TCP + UDP) ---

    #[test]
    fn tcp_connect_denied_by_default() {
        assert!(!socket_addr_permitted(
            SocketAddrUse::TcpConnect,
            false,
            false,
            false
        ));
    }

    #[test]
    fn tcp_connect_allowed_when_opted_in() {
        assert!(socket_addr_permitted(
            SocketAddrUse::TcpConnect,
            false,
            false,
            true
        ));
    }

    #[test]
    fn udp_connect_denied_by_default() {
        // E15: raw UDP egress was allowed unconditionally; now gated.
        assert!(!socket_addr_permitted(
            SocketAddrUse::UdpConnect,
            false,
            false,
            false
        ));
    }

    #[test]
    fn udp_connect_allowed_when_opted_in() {
        assert!(socket_addr_permitted(
            SocketAddrUse::UdpConnect,
            false,
            false,
            true
        ));
    }

    #[test]
    fn udp_outgoing_datagram_denied_by_default() {
        assert!(!socket_addr_permitted(
            SocketAddrUse::UdpOutgoingDatagram,
            false,
            false,
            false
        ));
    }

    #[test]
    fn udp_outgoing_datagram_allowed_when_opted_in() {
        assert!(socket_addr_permitted(
            SocketAddrUse::UdpOutgoingDatagram,
            false,
            false,
            true
        ));
    }

    #[test]
    fn opt_in_does_not_widen_bind() {
        // The raw-egress opt-in must not relax the bind posture.
        assert!(!socket_addr_permitted(
            SocketAddrUse::UdpBind,
            false,
            true,
            true
        ));
        assert!(!socket_addr_permitted(
            SocketAddrUse::UdpBind,
            true,
            false,
            true
        ));
        assert!(!socket_addr_permitted(
            SocketAddrUse::TcpBind,
            false,
            true,
            true
        ));
    }

    // --- is_raw_egress: only the outbound-egress uses drive the warn-once ---

    #[test]
    fn is_raw_egress_classifies_only_egress() {
        assert!(is_raw_egress(SocketAddrUse::TcpConnect));
        assert!(is_raw_egress(SocketAddrUse::UdpConnect));
        assert!(is_raw_egress(SocketAddrUse::UdpOutgoingDatagram));
        assert!(!is_raw_egress(SocketAddrUse::TcpBind));
        assert!(!is_raw_egress(SocketAddrUse::UdpBind));
    }
}
