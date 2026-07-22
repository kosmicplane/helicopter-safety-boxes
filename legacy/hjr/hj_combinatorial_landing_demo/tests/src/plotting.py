"""Comprehensive figures and matrix exports for the landing demonstration."""
from __future__ import annotations

from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage.measure import marching_cubes

from .hj_reachability import rth_largest


ZONE_COLORS = ["#E63946", "#2A9D8F", "#3A86FF", "#9B5DE5"]


def set_style(dpi: int = 220) -> None:
    plt.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": dpi,
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.labelsize": 9.5,
        "legend.fontsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def _save(fig, directory: Path, name: str, pdf: bool) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", bbox_inches="tight", facecolor="white")
    if pdf:
        fig.savefig(directory / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _set_3d(ax, world, title: str) -> None:
    # Matplotlib may normalize the array passed to set_box_aspect in-place.
    # Always work on a copy so the physical world dimensions are never mutated.
    size = np.asarray(world.size, dtype=float).copy()
    ax.set_xlim(0, float(size[0])); ax.set_ylim(0, float(size[1])); ax.set_zlim(0, float(size[2]))
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.set_title(title, fontweight="bold")
    ax.view_init(elev=24, azim=-60)
    try:
        ax.set_box_aspect(tuple(float(v) for v in size))
    except Exception:
        pass


def _draw_primitives(ax, world, include_hidden: bool, alpha: float = 0.20) -> None:
    for item in world.visible_primitives + ([world.hidden_hazard] if include_hidden else []):
        if item["kind"] == "sphere":
            c = np.asarray(item["center"]); r = item["radius"]
            u = np.linspace(0, 2 * np.pi, 24); v = np.linspace(0, np.pi, 14)
            X = c[0] + r * np.outer(np.cos(u), np.sin(v))
            Y = c[1] + r * np.outer(np.sin(u), np.sin(v))
            Z = c[2] + r * np.outer(np.ones_like(u), np.cos(v))
            ax.plot_surface(X, Y, Z, alpha=alpha, linewidth=0, color="#6C757D" if item is not world.hidden_hazard else "#D90429")
        elif item["kind"] == "cylinder":
            c = np.asarray(item["center"]); r = item["radius"]; h = item["height"]
            theta, zz = np.meshgrid(np.linspace(0, 2*np.pi, 24), np.linspace(-0.5, 0.5, 12))
            ax.plot_surface(c[0]+r*np.cos(theta), c[1]+r*np.sin(theta), c[2]+h*zz, alpha=alpha, linewidth=0, color="#495057")
        elif item["kind"] == "box":
            c = np.asarray(item["center"]); s = np.asarray(item["size"])
            xs = [c[0]-s[0]/2, c[0]+s[0]/2]; ys=[c[1]-s[1]/2,c[1]+s[1]/2]; zs=[c[2]-s[2]/2,c[2]+s[2]/2]
            corners=np.array([[x,y,z] for x in xs for y in ys for z in zs])
            edges=[(0,1),(0,2),(0,4),(3,1),(3,2),(3,7),(5,1),(5,4),(5,7),(6,2),(6,4),(6,7)]
            for a,b in edges: ax.plot(*zip(corners[a], corners[b]), color="#6C757D", alpha=0.8, lw=1.0)


def _draw_zones(ax, world) -> None:
    for i, zone in enumerate(world.landing_zones):
        ax.scatter(*zone.center, marker="D", s=75, color=ZONE_COLORS[i], edgecolor="white", label=zone.name)
    ax.scatter(*world.start, marker="o", s=85, color="#F4A261", edgecolor="white", label="Start A")
    ax.scatter(*world.science_waypoint, marker="*", s=120, color="#FFD166", edgecolor="black", linewidth=0.5, label="Science waypoint B")


def fig_architecture(out: Path, pdf: bool) -> None:
    fig, ax = plt.subplots(figsize=(15, 6.2)); ax.axis("off")
    ax.set_title("Four-Zone Combinatorial HJ Reachability + Poisson-CBF Safety Architecture", fontsize=16, fontweight="bold")
    blocks = [
        (0.08,0.62,"Occupancy / perception","initial map + new LZ hazard"),
        (0.29,0.62,"Poisson Safety Box",r"$O\rightarrow h_P,\nabla h_P,\nabla^2h_P$"),
        (0.51,0.62,"HJ reachability",r"$V_j=v_{max}(-\tau)-D_j$, $j=1..4$"),
        (0.73,0.62,"Combinatorial QP",r"active + 4 contingency + Poisson constraints"),
        (0.92,0.62,"Velocity setpoint",r"$u_{safe}\rightarrow$ PX4 tracker"),
        (0.29,0.26,"Nominal mission planner","A → science region B → provisional LZ"),
        (0.62,0.26,"Target manager",r"reject LZ-1, switch with $\tau_1\leftarrow\tau_2$"),
    ]
    for x,y,title,sub in blocks:
        w=0.16 if x<0.9 else 0.13
        ax.add_patch(plt.Rectangle((x-w/2,y-0.075),w,0.14,transform=ax.transAxes,fc="#F8F9FA",ec="#1D3557",lw=1.4))
        ax.text(x,y+0.016,title,ha="center",va="center",transform=ax.transAxes,fontweight="bold",color="#1D3557")
        ax.text(x,y-0.035,sub,ha="center",va="center",transform=ax.transAxes,fontsize=8.4)
    arrows=[((.16,.62),(.21,.62)),((.37,.62),(.43,.62)),((.59,.62),(.65,.62)),((.81,.62),(.855,.62)),((.29,.33),(.67,.55)),((.62,.33),(.71,.55)),((.08,.55),(.57,.31))]
    for a,b in arrows: ax.annotate("",xy=b,xytext=a,xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->",lw=1.4,color="#1D3557"))
    ax.text(.5,.08,"Requirement: p=4 landing zones, at least r=2 must remain reachable until target confirmation/switching.",ha="center",transform=ax.transAxes,fontsize=11,fontweight="bold")
    _save(fig,out,"fig01_architecture",pdf)


def fig_world_trajectory(art, out: Path, pdf: bool) -> None:
    world, log = art.world, art.log
    pos = log["position"]; active = log["active_index"].astype(int); discovered=log["discovered"].astype(bool)
    fig=plt.figure(figsize=(16,7.5))
    ax=fig.add_subplot(121,projection="3d"); _draw_primitives(ax,world,False,0.20); _draw_zones(ax,world); _set_3d(ax,world,"Prior world and four landing zones"); ax.legend(loc="upper left")
    ax=fig.add_subplot(122,projection="3d"); _draw_primitives(ax,world,True,0.20); _draw_zones(ax,world)
    for i in range(len(world.landing_zones)):
        mask=active==i
        if np.any(mask): ax.plot(pos[mask,0],pos[mask,1],pos[mask,2],color=ZONE_COLORS[i],lw=2.7,label=f"trajectory while targeting {world.landing_zones[i].name}")
    switch_idx=np.argmax(discovered) if np.any(discovered) else None
    if switch_idx is not None: ax.scatter(*pos[switch_idx],s=120,marker="X",color="black",label="hazard discovery / target switch")
    _set_3d(ax,world,"Filtered 3-D trajectory after map update"); ax.legend(loc="upper left")
    fig.suptitle("Mars-analog flight A→B with contingency-preserving landing diversion",fontsize=16,fontweight="bold")
    _save(fig,out,"fig02_world_and_3d_trajectory",pdf)


def _slice_indices(world):
    return [max(1,min(world.shape[2]-2,int(q*(world.shape[2]-1)))) for q in (0.10,0.30,0.52,0.75)]


def _matrix_slices(world, arrays, titles, cmap, out, name, pdf, contours=None):
    ks=_slice_indices(world); fig,axes=plt.subplots(len(arrays),4,figsize=(16,3.3*len(arrays)),constrained_layout=True)
    if len(arrays)==1: axes=axes[None,:]
    X,Y,_=world.mesh
    for row,(arr,row_title) in enumerate(zip(arrays,titles)):
        for col,k in enumerate(ks):
            ax=axes[row,col]; im=ax.imshow(arr[:,:,k].T,origin="lower",extent=[0,world.size[0],0,world.size[1]],cmap=cmap,interpolation="nearest" if arr.dtype==bool else "bilinear")
            if contours is not None: ax.contour(X[:,:,k],Y[:,:,k],contours[row][:,:,k].astype(float),levels=[0.5],colors="white",linewidths=.7)
            ax.set_title(f"{row_title}, z={world.axes[2][k]:.2f} m"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal"); fig.colorbar(im,ax=ax,fraction=.046)
    _save(fig,out,name,pdf)


def fig_occupancy_boundary(art,out,pdf):
    cmap=ListedColormap(["white","#264653"])
    _matrix_slices(art.world,[art.poisson_before.occupancy_mask,art.poisson_after.occupancy_mask],["Occupancy before discovery","Occupancy after LZ-1 hazard"],cmap,out,"fig03_occupancy_matrix_slices",pdf)
    bmap=ListedColormap(["white","#E76F51"])
    _matrix_slices(art.world,[art.poisson_before.boundary_mask,art.poisson_after.boundary_mask],["Dirichlet boundary before","Dirichlet boundary after"],bmap,out,"fig04_boundary_matrix_slices",pdf)


def fig_forcing_h(art,out,pdf):
    _matrix_slices(art.world,[art.poisson_before.forcing,art.poisson_after.forcing],["Poisson forcing before","Poisson forcing after"],"magma",out,"fig05_poisson_forcing_matrices",pdf,contours=[art.poisson_before.occupancy_mask,art.poisson_after.occupancy_mask])
    _matrix_slices(art.world,[art.poisson_before.h,art.poisson_after.h],["Safety function h before","Safety function h after"],"viridis",out,"fig06_poisson_h_matrices",pdf,contours=[art.poisson_before.occupancy_mask,art.poisson_after.occupancy_mask])


def fig_grad_hessian(art,out,pdf):
    world=art.world; k=_slice_indices(world)[2]; fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for row,(res,label) in enumerate([(art.poisson_before,"before"),(art.poisson_after,"after")]):
        h=res.h[:,:,k]; grad=res.grad_h[:,:,k,:]; gmag=np.linalg.norm(grad,axis=-1); H=res.hessian_h[:,:,k,:,:]; frob=np.linalg.norm(H,axis=(-2,-1)); lap=res.laplacian_h[:,:,k]
        for col,(arr,title,cmap) in enumerate([(gmag,r"$\|\nabla h\|$","inferno"),(frob,r"$\|\nabla^2 h\|_F$","plasma"),(lap,r"$\Delta h$","coolwarm")]):
            ax=axes[row,col]; im=ax.imshow(arr.T,origin="lower",extent=[0,world.size[0],0,world.size[1]],cmap=cmap,interpolation="bilinear"); ax.set_title(f"{title} {label}, z={world.axes[2][k]:.2f} m"); ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); fig.colorbar(im,ax=ax,fraction=.046)
            if col==0:
                skip=(slice(None,None,4),slice(None,None,4)); X,Y,_=world.mesh; ax.quiver(X[:,:,k][skip],Y[:,:,k][skip],grad[:,:,0][skip],grad[:,:,1][skip],color="white",scale=1.6,width=.0028)
    fig.suptitle("Poisson derivatives consumed by CBF/HOCBF layers",fontsize=15,fontweight="bold")
    _save(fig,out,"fig07_poisson_gradient_hessian_laplacian",pdf)


def fig_isosurfaces(art,out,pdf):
    world=art.world; fig=plt.figure(figsize=(15,7))
    for col,(res,label,hidden) in enumerate([(art.poisson_before,"before discovery",False),(art.poisson_after,"after discovery",True)],start=1):
        ax=fig.add_subplot(1,2,col,projection="3d"); h=res.h; positive=h[h>0]
        if positive.size:
            for frac,color,alpha in [(0.18,"#90BE6D",.15),(0.38,"#43AA8B",.20),(0.62,"#277DA1",.24)]:
                level=float(np.quantile(positive,frac))
                try:
                    verts,faces,_,_=marching_cubes(h,level=level,spacing=world.spacing); mesh=Poly3DCollection(verts[faces],alpha=alpha,linewidths=0); mesh.set_facecolor(color); ax.add_collection3d(mesh)
                except Exception: pass
        _draw_primitives(ax,world,hidden,.13); _draw_zones(ax,world); _set_3d(ax,world,f"3-D h isosurfaces {label}")
    fig.suptitle("Three-dimensional Poisson safety landscape",fontsize=16,fontweight="bold")
    _save(fig,out,"fig08_poisson_3d_isosurfaces",pdf)


def fig_reachability_fields(art,out,pdf):
    world=art.world; k=int(np.argmin(np.abs(world.axes[2]-3.0))); tau=-5.5
    fig,axes=plt.subplots(2,4,figsize=(17,7.5),constrained_layout=True)
    for j in range(4):
        for row,(fields,label) in enumerate([(art.reach_before,"before"),(art.reach_after,"after")]):
            V=fields[j].value_grid(tau)[:,:,k]; ax=axes[row,j]; finite=V>-900; data=np.where(finite,V,np.nan); im=ax.imshow(data.T,origin="lower",extent=[0,world.size[0],0,world.size[1]],cmap="coolwarm",vmin=-8,vmax=8,interpolation="bilinear"); ax.contour(world.mesh[0][:,:,k],world.mesh[1][:,:,k],V,levels=[0],colors="black",linewidths=1.2); ax.scatter(*world.landing_zones[j].center[:2],marker="D",s=55,color=ZONE_COLORS[j],edgecolor="white"); ax.set_title(f"{world.landing_zones[j].name} V_j {label}"); ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); fig.colorbar(im,ax=ax,fraction=.046)
    fig.suptitle(r"Four Hamilton–Jacobi reach-avoid value matrices $V_j(p,\tau_2)$; black contour is $V_j=0$",fontsize=15,fontweight="bold")
    _save(fig,out,"fig09_hj_value_matrices_four_zones",pdf)


def fig_geodesic_and_pivot(art,out,pdf):
    world=art.world; k=int(np.argmin(np.abs(world.axes[2]-3.0))); tau=-5.5; r=2
    fig,axes=plt.subplots(2,3,figsize=(15.5,9),constrained_layout=True)
    for row,(fields,label) in enumerate([(art.reach_before,"before discovery"),(art.reach_after,"after discovery")]):
        values=np.stack([f.value_grid(tau) for f in fields],axis=-1); sorted_v=np.sort(values,axis=-1)[...,::-1]; pivot=sorted_v[...,r-1]; count=np.sum(values>=0,axis=-1)
        distance_stack = np.stack([np.where(f.finite_mask, f.distance, np.inf) for f in fields], axis=-1)
        dist = np.min(distance_stack, axis=-1)
        dist[~np.isfinite(dist)] = np.nan
        for col,(arr,title,cmap) in enumerate([(dist[:,:,k],"Nearest obstacle-aware target distance","viridis"),(pivot[:,:,k],r"Combinatorial pivot $\tilde h$ (2nd largest)","coolwarm"),(count[:,:,k],"Number of reachable landing zones","viridis")]):
            ax=axes[row,col]; im=ax.imshow(arr.T,origin="lower",extent=[0,world.size[0],0,world.size[1]],cmap=cmap,interpolation="bilinear"); ax.set_title(f"{title}\n{label}"); ax.set_aspect("equal"); ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); fig.colorbar(im,ax=ax,fraction=.046)
            if col==1: ax.contour(world.mesh[0][:,:,k],world.mesh[1][:,:,k],pivot[:,:,k],levels=[0],colors="black",linewidths=1.2)
    fig.suptitle("Geodesic reachability and r=2 combinatorial set",fontsize=16,fontweight="bold")
    _save(fig,out,"fig10_geodesic_distance_pivot_reachable_count",pdf)


def fig_time_reachability(art,out,pdf):
    log=art.log; t=log["time"]; values=np.vstack(log["zone_values"]); active=log["active_index"].astype(int)
    fig,axes=plt.subplots(3,1,figsize=(14,9),sharex=True,constrained_layout=True)
    display_values = values.copy()
    # Rejected targets are encoded internally with a large negative sentinel.
    # Stop drawing that curve after rejection so the remaining margins stay readable.
    display_values[display_values < -100.0] = np.nan
    for j in range(4):
        axes[0].plot(t,display_values[:,j],color=ZONE_COLORS[j],label=f"V{j+1} ({art.world.landing_zones[j].name})")
    discovery_indices = np.flatnonzero(log["discovered"].astype(bool))
    if discovery_indices.size:
        t_discovery = float(t[discovery_indices[0]])
        for ax in axes:
            ax.axvline(t_discovery,color="#D90429",ls=":",lw=1.5)
        axes[0].annotate("LZ-1 rejected / divert",xy=(t_discovery,0.0),xytext=(t_discovery+0.4,5.0),arrowprops=dict(arrowstyle="->",color="#D90429"),color="#D90429")
    axes[0].axhline(0,color="black",ls="--",lw=1); axes[0].set_ylabel(r"$V_j(p,\tau_2)$"); axes[0].set_title("Contingency reachability values"); axes[0].legend(ncol=2)
    axes[1].plot(t,log["pivot_value"],label=r"pivot $\tilde h$",lw=2.3); axes[1].axhline(0,color="black",ls="--",lw=1); axes[1].set_ylabel("pivot value"); axes[1].set_title("r=2 guarantee: pivot must remain nonnegative")
    axes[2].step(t,log["reachable_count"],where="post",label="reachable count"); axes[2].step(t,active+1,where="post",label="active target index"); axes[2].axhline(2,color="black",ls="--",label="required r=2"); axes[2].set_ylabel("count / index"); axes[2].set_xlabel("time [s]"); axes[2].legend(); axes[2].set_title("Target switch and number of retained contingencies")
    _save(fig,out,"fig11_reachability_time_histories",pdf)


def fig_safety_control(art,out,pdf):
    log=art.log; t=log["time"]; un=np.vstack(log["u_nom"]); uc=np.vstack(log["u_cbf_only"]); us=np.vstack(log["u_safe"])
    fig,axes=plt.subplots(4,1,figsize=(14,11),sharex=True,constrained_layout=True)
    margin = float(log["poisson_h"][0] - log["poisson_h_margin_value"][0])
    axes[0].plot(t,log["poisson_h"],label="raw Poisson h"); axes[0].axhline(margin,color="#D62828",ls="--",label=f"configured h margin={margin:.3f}"); axes[0].axhline(0,color="black",ls=":"); axes[0].set_ylabel("h"); axes[0].set_title("Poisson safety value and configured safety buffer"); axes[0].legend()
    axes[1].plot(t,log["poisson_residual"],label="Poisson CBF residual"); axes[1].plot(t,log["active_hj_residual"],label="active HJ residual"); axes[1].plot(t,np.min(np.vstack(log["contingency_residuals"]),axis=1),label="minimum contingency residual"); axes[1].axhline(0,color="black",ls="--"); axes[1].set_ylabel("residual"); axes[1].legend(ncol=3); axes[1].set_title("All safety/reachability inequalities; nonnegative is feasible")
    labels=["x","y","z"]
    for d in range(3): axes[2].plot(t,un[:,d],ls="--",alpha=.55,label=f"nom {labels[d]}"); axes[2].plot(t,us[:,d],lw=1.8,label=f"safe {labels[d]}")
    axes[2].set_ylabel("velocity [m/s]"); axes[2].legend(ncol=3); axes[2].set_title("Nominal versus unified safe command")
    axes[3].plot(t,log["correction_norm"],label=r"$\|u_{safe}-u_{nom}\|$"); axes[3].plot(t,np.linalg.norm(uc-un,axis=1),label="CBF-only correction",alpha=.8); axes[3].set_ylabel("correction [m/s]"); axes[3].set_xlabel("time [s]"); axes[3].legend(); axes[3].set_title("Intervention effort")
    _save(fig,out,"fig12_poisson_h_cbf_hj_control_histories",pdf)


def fig_qp_diagnostics(art,out,pdf):
    log=art.log;t=log["time"]
    fig,axes=plt.subplots(3,1,figsize=(13,8.5),sharex=True,constrained_layout=True)
    axes[0].plot(t,log["omega_active"],label=r"$\omega_1$ active"); axes[0].plot(t,log["omega_contingency"],label=r"$\omega_2$ contingency"); axes[0].legend(); axes[0].set_ylabel("auxiliary variable"); axes[0].set_title("Paper relaxation variables")
    axes[1].plot(t,log["qp_solve_ms"]); axes[1].set_ylabel("solve time [ms]"); axes[1].set_title("Online unified QP solve time")
    axes[2].step(t,log["qp_success"],where="post"); axes[2].set_ylim(-.05,1.05); axes[2].set_ylabel("success"); axes[2].set_xlabel("time [s]"); axes[2].set_title("Numerical QP feasibility status")
    _save(fig,out,"fig13_qp_relaxation_and_timing",pdf)


def fig_dashboard(art,out,pdf):
    world,log=art.world,art.log; pos=log["position"]; t=log["time"]; values=np.vstack(log["zone_values"])
    fig=plt.figure(figsize=(18,11)); gs=fig.add_gridspec(3,3,hspace=.35,wspace=.30)
    ax=fig.add_subplot(gs[0,0],projection="3d"); _draw_primitives(ax,world,True,.13); _draw_zones(ax,world); ax.plot(pos[:,0],pos[:,1],pos[:,2],color="#1D3557",lw=2.3); _set_3d(ax,world,"3-D mission")
    k=_slice_indices(world)[1]
    ax=fig.add_subplot(gs[0,1]); im=ax.imshow(art.poisson_after.occupancy_mask[:,:,k].T,origin="lower",extent=[0,world.size[0],0,world.size[1]],cmap=ListedColormap(["white","#264653"])); ax.set_title("Updated occupancy"); ax.set_aspect("equal")
    ax=fig.add_subplot(gs[0,2]); im=ax.imshow(art.poisson_after.h[:,:,k].T,origin="lower",extent=[0,world.size[0],0,world.size[1]],cmap="viridis"); ax.set_title("Updated Poisson h"); ax.set_aspect("equal"); fig.colorbar(im,ax=ax,fraction=.046)
    ax=fig.add_subplot(gs[1,0]);
    for j in range(4): ax.plot(t,values[:,j],color=ZONE_COLORS[j],label=f"V{j+1}")
    ax.axhline(0,color="black",ls="--"); ax.set_title("Four reachability values"); ax.legend(ncol=2)
    ax=fig.add_subplot(gs[1,1]); ax.plot(t,log["pivot_value"],label="pivot"); ax.axhline(0,color="black",ls="--"); ax.set_title("r=2 pivot certificate")
    ax=fig.add_subplot(gs[1,2]); ax.step(t,log["reachable_count"],where="post",label="reachable"); ax.step(t,log["active_index"]+1,where="post",label="active target"); ax.legend(); ax.set_title("Target switch")
    ax=fig.add_subplot(gs[2,0]); ax.plot(t,log["poisson_h"]); ax.axhline(0,color="black",ls="--"); ax.set_title("Poisson h(t)"); ax.set_xlabel("time [s]")
    ax=fig.add_subplot(gs[2,1]); ax.plot(t,log["correction_norm"]); ax.set_title("Unified-filter correction"); ax.set_xlabel("time [s]")
    ax=fig.add_subplot(gs[2,2]); ax.plot(t,log["qp_solve_ms"]); ax.set_title("QP solve time [ms]"); ax.set_xlabel("time [s]")
    fig.suptitle("Integrated 4-zone HJ + Poisson-CBF landing dashboard",fontsize=18,fontweight="bold")
    _save(fig,out,"fig14_integrated_dashboard",pdf)


def fig_hj_3d_reachable_sets(art, out, pdf):
    """Render the V_j=0 surfaces for all four targets before and after discovery."""
    world = art.world
    tau = -5.5
    fig = plt.figure(figsize=(18, 9))
    cases = [(art.reach_before, "before discovery", False), (art.reach_after, "after discovery", True)]
    for row, (fields, label, hidden) in enumerate(cases):
        for j, field in enumerate(fields):
            ax = fig.add_subplot(2, 4, row * 4 + j + 1, projection="3d")
            V = field.value_grid(tau)
            if np.nanmin(V) < 0.0 < np.nanmax(V):
                try:
                    verts, faces, _, _ = marching_cubes(V, level=0.0, spacing=world.spacing)
                    mesh = Poly3DCollection(verts[faces], alpha=0.22, linewidths=0.0)
                    mesh.set_facecolor(ZONE_COLORS[j])
                    ax.add_collection3d(mesh)
                except Exception:
                    pass
            _draw_primitives(ax, world, hidden, 0.08)
            ax.scatter(*world.landing_zones[j].center, marker="D", s=55,
                       color=ZONE_COLORS[j], edgecolor="white")
            _set_3d(ax, world, f"{world.landing_zones[j].name}\n{label}")
    fig.suptitle(r"3-D Hamilton–Jacobi backward reach-avoid boundaries $V_j(p,\tau)=0$ at $\tau=-5.5$ s",
                 fontsize=16, fontweight="bold")
    _save(fig, out, "fig15_hj_3d_reachable_sets", pdf)

def export_data(art, output_dir: Path) -> None:
    data=output_dir/"data"; data.mkdir(parents=True,exist_ok=True)
    art.poisson_before.save_npz(data/"poisson_before_discovery.npz"); art.poisson_before.save_summary_json(data/"poisson_before_summary.json")
    art.poisson_after.save_npz(data/"poisson_after_discovery.npz"); art.poisson_after.save_summary_json(data/"poisson_after_summary.json")
    np.savez_compressed(data/"reachability_before.npz",**{f"distance_LZ{j+1}":f.distance for j,f in enumerate(art.reach_before)},**{f"grad_distance_LZ{j+1}":f.grad_distance for j,f in enumerate(art.reach_before)})
    np.savez_compressed(data/"reachability_after.npz",**{f"distance_LZ{j+1}":f.distance for j,f in enumerate(art.reach_after)},**{f"grad_distance_LZ{j+1}":f.grad_distance for j,f in enumerate(art.reach_after)})
    log=art.log; pos=np.vstack(log["position"]); un=np.vstack(log["u_nom"]); us=np.vstack(log["u_safe"]); vals=np.vstack(log["zone_values"])
    headers=["time","x","y","z","u_nom_x","u_nom_y","u_nom_z","u_safe_x","u_safe_y","u_safe_z","active_index","poisson_h","pivot_value","reachable_count","V1","V2","V3","V4","poisson_residual","active_hj_residual","omega_active","omega_contingency","qp_solve_ms","qp_success"]
    with open(data/"simulation_log.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(headers)
        for i in range(len(log["time"])):
            w.writerow([log["time"][i],*pos[i],*un[i],*us[i],int(log["active_index"][i]),log["poisson_h"][i],log["pivot_value"][i],int(log["reachable_count"][i]),*vals[i],log["poisson_residual"][i],log["active_hj_residual"][i],log["omega_active"][i],log["omega_contingency"][i],log["qp_solve_ms"][i],int(log["qp_success"][i])])


def generate_all_figures(art, output_dir: str | Path, plot_config: dict) -> None:
    output_dir=Path(output_dir); figures=output_dir/"figures"; set_style(int(plot_config.get("dpi",220))); pdf=bool(plot_config.get("save_pdf",False))
    fig_architecture(figures,pdf); fig_world_trajectory(art,figures,pdf); fig_occupancy_boundary(art,figures,pdf); fig_forcing_h(art,figures,pdf); fig_grad_hessian(art,figures,pdf); fig_isosurfaces(art,figures,pdf); fig_reachability_fields(art,figures,pdf); fig_geodesic_and_pivot(art,figures,pdf); fig_time_reachability(art,figures,pdf); fig_safety_control(art,figures,pdf); fig_qp_diagnostics(art,figures,pdf); fig_dashboard(art,figures,pdf); fig_hj_3d_reachable_sets(art,figures,pdf); export_data(art,output_dir)
