import { VisionFeed } from "@/components/VisionFeed";

export default function Home() {
  return (
    <main className="min-h-screen bg-background flex flex-col items-center justify-center p-4">
      <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-primary via-secondary to-primary opacity-50" />
      <VisionFeed />
      <footer className="mt-12 text-muted-foreground text-sm font-medium opacity-60">
        Vision Canvas &copy; {new Date().getFullYear()} — Advanced Browser Visualization
      </footer>
    </main>
  );
}