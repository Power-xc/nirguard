import Link from "next/link";

export default function NotFound() {
  return (
    <main className="not-found-page">
      <section className="not-found-panel">
        <p className="eyebrow">NIRGuard Secure Viewer</p>
        <h1>Page not found</h1>
        <p>The requested console view is not available.</p>
        <Link className="button primary" href="/">
          Return to console
        </Link>
      </section>
    </main>
  );
}
