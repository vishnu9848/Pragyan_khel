'use client';

import dynamic from 'next/dynamic';
import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Sparkles, ArrowRight, Zap, Shield, Target } from 'lucide-react';

const VisionFeed = dynamic(
  () => import("@/components/VisionFeed").then((mod) => mod.VisionFeed),
  { 
    ssr: false,
    loading: () => (
      <div className="flex flex-col items-center justify-center min-h-[500px] gap-4 bg-white rounded-3xl border border-slate-100 shadow-sm">
        <div className="w-10 h-10 border-4 border-primary border-t-transparent rounded-full animate-spin" />
        <p className="text-muted-foreground font-mono text-[10px] tracking-[0.2em] uppercase">Syncing Neural Core...</p>
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
    <main className="min-h-screen bg-background flex flex-col items-center justify-center p-6 relative overflow-hidden selection:bg-primary/20 selection:text-primary">
      {/* Soft Decorative Ambient Gradients */}
      <div className="absolute top-[-20%] left-[-10%] w-[60%] h-[60%] bg-primary/5 rounded-full blur-[120px] pointer-events-none animate-pulse" />
      <div className="absolute bottom-[-20%] right-[-10%] w-[60%] h-[60%] bg-blue-500/5 rounded-full blur-[120px] pointer-events-none animate-pulse" style={{ animationDelay: '3s' }} />
      
      <div className="relative z-10 w-full max-w-6xl flex flex-col items-center">
        {!isStarted ? (
          <div className="flex flex-col items-center text-center space-y-12 animate-fade-in py-16">
            <div className="space-y-8">
              <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-primary/10 border border-primary/20 text-primary text-[10px] font-bold uppercase tracking-[0.3em] mb-2">
                <Sparkles className="w-3.5 h-3.5" /> Intelligence Suite 3.5
              </div>
              <h1 className="text-6xl md:text-8xl font-black tracking-tight text-slate-900">
                Vision <span className="text-primary italic">Canvas</span>
              </h1>
              <div className="max-w-2xl mx-auto space-y-6">
                <p className="text-xl md:text-2xl text-slate-500 font-medium leading-relaxed italic">
                  "Seeing the unseen through the lens of artificial intelligence."
                </p>
                <div className="w-12 h-0.5 bg-slate-200 mx-auto rounded-full" />
                <p className="text-[10px] font-mono text-slate-400 uppercase tracking-[0.4em]">
                  — Precision Vision Directive 01
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 w-full max-w-3xl">
               <div className="flex flex-col items-center gap-3 p-6 rounded-2xl bg-white border border-slate-100 shadow-sm transition-transform hover:scale-[1.02]">
                 <Target className="w-6 h-6 text-primary" />
                 <span className="text-[10px] font-bold uppercase tracking-widest text-slate-900">Precision Lock</span>
               </div>
               <div className="flex flex-col items-center gap-3 p-6 rounded-2xl bg-white border border-slate-100 shadow-sm transition-transform hover:scale-[1.02]">
                 <Zap className="w-6 h-6 text-primary" />
                 <span className="text-[10px] font-bold uppercase tracking-widest text-slate-900">Neural Logic</span>
               </div>
               <div className="flex flex-col items-center gap-3 p-6 rounded-2xl bg-white border border-slate-100 shadow-sm transition-transform hover:scale-[1.02]">
                 <Shield className="w-6 h-6 text-primary" />
                 <span className="text-[10px] font-bold uppercase tracking-widest text-slate-900">Secure Link</span>
               </div>
            </div>

            <Button 
              onClick={() => setIsStarted(true)}
              className="group h-16 px-12 rounded-2xl bg-primary text-white hover:bg-primary/90 transition-all hover:scale-[1.05] active:scale-95 text-base font-bold uppercase tracking-[0.2em] shadow-lg shadow-primary/20"
            >
              Initialize System
              <ArrowRight className="ml-3 w-5 h-5 group-hover:translate-x-1.5 transition-transform" />
            </Button>
          </div>
        ) : (
          <div className="w-full animate-scale-in">
            <VisionFeed />
          </div>
        )}
      </div>

      <footer className="mt-16 text-slate-400 text-[10px] font-mono tracking-[0.3em] uppercase flex items-center gap-4 relative z-10">
        <span>Vision Canvas &copy; {year || '...'}</span>
        <div className="w-1 h-1 rounded-full bg-slate-300" />
        <span className="text-primary font-bold">PRO.CORE_V3</span>
      </footer>
    </main>
  );
}
