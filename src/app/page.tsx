import { VisionFeed } from "@/components/VisionFeed";

export default function Home() {
  return (
    <main className="min-h-screen bg-[#020202] flex flex-col items-center justify-center p-4 relative overflow-hidden">
      {/* Cinematic Background Gradients */}
      <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-primary/20 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-accent/10 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 pointer-events-none mix-blend-overlay" />
      
      <div className="relative z-10 w-full max-w-6xl">
        <VisionFeed />
      </div>

      <footer className="mt-12 text-muted-foreground/40 text-xs font-mono tracking-widest uppercase">
        Vision Canvas &copy; {new Date().getFullYear()} // Neural Processing Engine v2.5
      </footer>
    </main>
  );
}