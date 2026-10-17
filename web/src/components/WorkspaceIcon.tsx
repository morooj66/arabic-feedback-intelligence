export function WorkspaceIcon({ name }: { name: string }) {
  const paths: Record<string, string> = {
    dashboard: "M3 10l9-7 9 7v10H3V10M9 20v-7h6v7",
    assignment: "M16 20v-2a4 4 0 00-4-4H6a4 4 0 00-4 4v2M16 4a4 4 0 010 8M22 20v-2a4 4 0 00-3-3.87M9 11a4 4 0 100-8 4 4 0 000 8",
    completed: "M8 4H5v17h14V4h-3M8 3h8v4H8V3M8 14l3 3 5-6",
    analytics: "M4 20h17M6 16v-5M12 16V4M18 16V8",
    messages: "M3 5h18v14H3V5M3 6l9 7 9-7",
    review: "M12 8v4l3 2M21 12a9 9 0 11-18 0 9 9 0 0118 0",
    menu: "M4 6h16M4 12h16M4 18h16",
    close: "M6 6l12 12M6 18L18 6",
    chevron: "M6 9l6 6 6-6",
    upload: "M12 16V3M7 8l5-5 5 5M4 15v6h16v-6",
  };
  return <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name] || paths.review} /></svg>;
}
