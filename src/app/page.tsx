import { VisionFeed } from "@/components/VisionFeed";

export default function Home() {
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
        <span>Vision Canvas &copy; {new Date().getFullYear()}</span>
        <div className="w-1 h-1 rounded-full bg-muted-foreground/30" />
        <span className="text-primary/50">Neural Processing Engine v3.0</span>
      </footer>
    </main>
  );
}