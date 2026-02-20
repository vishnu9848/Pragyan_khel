'use client';

import dynamic from 'next/dynamic';
import { useState, useEffect } from 'react';

// Use dynamic import with ssr: false to prevent hydration errors from 
// browser-only APIs and Radix UI auto-generated IDs.
const VisionFeed = dynamic(
  () => import("@/components/VisionFeed").then((mod) => mod.VisionFeed),
  { 
    ssr: false,
    loading: () => (
      <div className="flex flex-col items-center justify-center min-h-[400px] gap-4">
        <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin" />
        <p className="text-muted-foreground font-mono text-xs tracking-widest uppercase">Initializing Neural Core...</p>
      </div>
    )
  }
);

export default function Home() {
  const [year, setYear] = useState<number | null>(null);

  useEffect(() => {
    setYear(new Date().getFullYear());
  }, []);

  return (
    <main className="min-h-screen bg-background flex flex-col items-center justify-center p-4 relative overflow-hidden">
      {/* Dynamic Background Gradients */}
      <div className="absolute top-[-15%] left-[-10%] w-[50%] h-[50%] bg-primary/15 rounded-full blur-[140px] pointer-events-none animate-pulse" />
      <div className="absolute bottom-[-15%] right-[-10%] w-[50%] h-[50%] bg-accent/10 rounded-full blur-[140px] pointer-events-none animate-pulse" style={{ animationDelay: '2s' }} />
      <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-[0.15] pointer-events-none mix-blend-overlay" />
      
      <div className="relative z-10 w-full max-w-6xl">
        <VisionFeed />
      </div>

      <footer className="mt-12 text-muted-foreground/30 text-[10px] font-mono tracking-[0.3em] uppercase flex items-center gap-4">
        <span>Vision Canvas &copy; {year || '...'}</span>
        <div className="w-1 h-1 rounded-full bg-muted-foreground/30" />
        <span className="text-primary/50">Neural Processing Engine v3.1</span>
      </footer>
    </main>
  );
}
