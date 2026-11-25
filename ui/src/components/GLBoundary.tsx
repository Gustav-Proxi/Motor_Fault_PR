import { Component, type ReactNode } from "react";

/** Renders `fallback` if a WebGL view throws (context creation failed or was lost). */
export class GLBoundary extends Component<{ fallback: ReactNode; children: ReactNode }, { failed: boolean }> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  componentDidCatch(err: unknown) {
    console.warn("3-D view unavailable, using the 2-D fallback:", err);
  }

  render() {
    return this.state.failed ? this.props.fallback : this.props.children;
  }
}
