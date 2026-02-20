'use client';

import dynamic from 'next/dynamic';
import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Sparkles, ArrowRight, Zap, Shield, Target } from 'lucide-react';

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
  const [isStarted, setIsStarted] = useState(false);

  useEffect(() => {
    setYear(new Date().getFullYear());
  }, []);

  return (
    <main className="min-h-screen bg-background flex flex-col items-center justify-center p-4 relative overflow-hidden">
      {/* Dynamic Background Gradients */}
      <div className="absolute top-[-15%] left-[-10%] w-[50%] h-[50%] bg-primary/15 rounded-full blur-[140px] pointer-events-none animate-pulse" />
      <div className="absolute bottom-[-15%] right-[-10%] w-[50%] h-[50%] bg-accent/10 rounded-full blur-[140px] pointer-events-none animate-pulse" style={{ animationDelay: '2s' }} />
      <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-[0.15] pointer-events-none mix-blend-overlay" />
      
      <div className="relative z-10 w-full max-w-6xl flex flex-col items-center">
        {!isStarted ? (
          <div className="flex flex-col items-center text-center space-y-12 animate-fade-in py-20">
            <div className="space-y-6">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-primary/10 border border-primary/20 text-primary text-[10px] font-black uppercase tracking-[0.3em] mb-4">
                <Sparkles className="w-3 h-3" /> System Version 3.1.0
              </div>
              <h1 className="text-7xl md:text-9xl font-black tracking-tighter text-white">
                Vision <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary via-accent to-primary animate-pulse">Canvas</span>
              </h1>
              <div className="max-w-xl mx-auto space-y-4">
                <p className="text-xl md:text-2xl text-muted-foreground font-medium italic leading-relaxed">
                  "Vision is the art of seeing what is invisible to others."
                </p>
                <p className="text-xs font-mono text-muted-foreground/40 uppercase tracking-[0.4em]">
                  — Neural Processing Directive 01
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 w-full max-w-3xl opacity-50 group-hover:opacity-100 transition-opacity">
               <div className="flex flex-col items-center gap-2 p-4 rounded-2xl bg-white/5 border border-white/5">
                 <Target className="w-5 h-5 text-accent" />
                 <span className="text-[10px] font-bold uppercase tracking-widest">Precision Lock</span>
               </div>
               <div className="flex flex-col items-center gap-2 p-4 rounded-2xl bg-white/5 border border-white/5">
                 <Zap className="w-5 h-5 text-primary" />
                 <span className="text-[10px] font-bold uppercase tracking-widest">Neural Logic</span>
               </div>
               <div className="flex flex-col items-center gap-2 p-4 rounded-2xl bg-white/5 border border-white/5">
                 <Shield className="w-5 h-5 text-white/40" />
                 <span className="text-[10px] font-bold uppercase tracking-widest">Secure Link</span>
               </div>
            </div>

            <Button 
              onClick={() => setIsStarted(true)}
              className="group h-20 px-12 rounded-[2rem] bg-white text-black hover:bg-white/90 transition-all hover:scale-[1.05] active:scale-95 text-lg font-black uppercase tracking-[0.2em] shadow-[0_0_50px_rgba(255,255,255,0.2)]"
            >
              Initialize System
              <ArrowRight className="ml-4 w-6 h-6 group-hover:translate-x-2 transition-transform" />
            </Button>
          </div>
        ) : (
          <div className="w-full animate-scale-in">
            <VisionFeed />
          </div>
        )}
      </div>

      <footer className="mt-12 text-muted-foreground/30 text-[10px] font-mono tracking-[0.3em] uppercase flex items-center gap-4 relative z-10">
        <span>Vision Canvas &copy; {year || '...'}</span>
        <div className="w-1 h-1 rounded-full bg-muted-foreground/30" />
        <span className="text-primary/50">Core.v3_Secure</span>
      </footer>
    </main>
  );
}
