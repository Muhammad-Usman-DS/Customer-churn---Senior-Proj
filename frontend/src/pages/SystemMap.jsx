// The system map is a standalone page (public/embed/system-map.html) with its own styles and
// script. It is framed rather than ported to React, and given the full viewport below the
// navbar, because its code sheet and full-screen mode rely on position: fixed and 100vh.
export default function SystemMap() {
  return (
    <iframe
      src="/embed/system-map.html"
      title="ChurnIQ System Map"
      allow="fullscreen"
      className="block w-full h-[calc(100vh-4rem)] h-[calc(100dvh-4rem)] border-0"
    />
  );
}
