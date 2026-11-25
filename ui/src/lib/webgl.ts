// WebGL availability. The 3-D views fall back to 2-D canvases when it is missing (old GPUs,
// blocklisted drivers, some VMs and remote desktops) or when ?webgl=0 is in the URL.

let cached: boolean | null = null;

export function hasWebGL(): boolean {
  if (cached !== null) return cached;
  try {
    if (new URLSearchParams(window.location.search).get("webgl") === "0") return (cached = false);
    const c = document.createElement("canvas");
    cached = !!(c.getContext("webgl2") || c.getContext("webgl"));
  } catch {
    cached = false;
  }
  return cached;
}
