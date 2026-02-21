'use client';

import dynamic from 'next/dynamic';
import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Eye, Lock, Mail, LogOut } from 'lucide-react';

const DEMO_EMAIL = 'demo@visionlock.com';
const DEMO_PASSWORD = 'VisionLock@123';

const VisionFeed = dynamic(
  () => import("@/components/VisionFeed").then((mod) => mod.VisionFeed),
  {
    ssr: false,
    loading: () => (
      <div className="flex flex-col items-center justify-center min-h-[500px] gap-4 glass-panel rounded-2xl border border-white/10">
        <div className="w-10 h-10 border-2 border-primary/60 border-t-primary rounded-full animate-spin" />
        <p className="text-muted-foreground text-[10px] font-medium tracking-[0.2em] uppercase">Syncing Neural Core...</p>
      </div>
    )
  }
);

export default function Home() {
  const [year, setYear] = useState<number | null>(null);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loginError, setLoginError] = useState('');
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setYear(new Date().getFullYear());
    setMounted(true);
  }, []);

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    setLoginError('');
    const trimmedEmail = email.trim().toLowerCase();
    if (trimmedEmail === DEMO_EMAIL && password === DEMO_PASSWORD) {
      setIsAuthenticated(true);
    } else {
      setLoginError('Invalid email or password. Use the demo credentials to sign in.');
    }
  };

  const handleLogout = () => {
    setIsAuthenticated(false);
    setEmail('');
    setPassword('');
    setLoginError('');
  };

  if (!mounted) {
    return (
      <main className="min-h-screen bg-background flex items-center justify-center">
        <div className="w-10 h-10 border-2 border-primary/60 border-t-primary rounded-full animate-spin" />
      </main>
    );
  }

  if (!isAuthenticated) {
    return (
      <main className="min-h-screen bg-background flex flex-col items-center justify-center p-6 relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,hsl(var(--primary)/0.15),transparent)]" />
        <div className="absolute inset-0 bg-[linear-gradient(to_right,hsl(var(--border)/0.4)_1px,transparent_1px),linear-gradient(to_bottom,hsl(var(--border)/0.4)_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_50%,black,transparent)]" />
        <div className="absolute top-[-20%] right-[-10%] w-[60%] h-[60%] bg-primary/10 rounded-full blur-[120px] pointer-events-none" />
        <div className="absolute bottom-[-20%] left-[-10%] w-[50%] h-[50%] bg-secondary/10 rounded-full blur-[100px] pointer-events-none" />

        <div className="relative z-10 w-full max-w-md">
          <div className="text-center mb-10 animate-fade-in">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-panel text-primary text-[10px] font-bold uppercase tracking-[0.3em] mb-6 border-primary/20">
              <Lock className="w-3.5 h-3.5" /> Secure Access
            </div>
            <h1 className="font-heading text-5xl md:text-6xl font-extrabold tracking-tight text-foreground mb-3">
              <span className="text-gradient">VisionLock</span>
            </h1>
            <p className="text-muted-foreground font-medium text-lg max-w-sm mx-auto">
              One subject in focus. Everything else fades.
            </p>
            <div className="mt-6 flex items-center justify-center gap-4 text-[10px] font-medium text-muted-foreground uppercase tracking-widest">
              <span>Cinematic portrait mode</span>
              <span className="w-1.5 h-1.5 rounded-full bg-primary/60" />
              <span>AI-powered</span>
            </div>
          </div>

          <div className="glass-dark rounded-2xl border border-white/10 shadow-2xl p-8 animate-scale-in emerald-glow">
            <form onSubmit={handleLogin} className="space-y-6">
              <div className="space-y-2">
                <Label htmlFor="email" className="text-xs font-bold uppercase tracking-widest text-muted-foreground flex items-center gap-2">
                  <Mail className="w-3.5 h-3.5 text-primary" /> Email
                </Label>
                <Input
                  id="email"
                  type="email"
                  placeholder="you@example.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="h-12 rounded-xl bg-white/5 border-white/10 text-foreground placeholder:text-muted-foreground focus-visible:ring-primary"
                  autoComplete="email"
                  required
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="password" className="text-xs font-bold uppercase tracking-widest text-muted-foreground flex items-center gap-2">
                  <Lock className="w-3.5 h-3.5 text-primary" /> Password
                </Label>
                <Input
                  id="password"
                  type="password"
                  placeholder="••••••••"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="h-12 rounded-xl bg-white/5 border-white/10 text-foreground placeholder:text-muted-foreground focus-visible:ring-primary"
                  autoComplete="current-password"
                  required
                />
              </div>
              {loginError && (
                <p className="text-sm text-destructive bg-destructive/10 border border-destructive/20 rounded-xl px-4 py-3 flex items-center gap-2">
                  <Eye className="w-4 h-4 shrink-0" /> {loginError}
                </p>
              )}
              <Button
                type="submit"
                className="w-full h-14 rounded-xl bg-primary text-primary-foreground hover:bg-primary/90 text-base font-bold uppercase tracking-widest shadow-lg emerald-glow"
              >
                Unlock VisionLock
              </Button>
            </form>
            <p className="mt-6 text-center text-xs text-muted-foreground">
              Demo: <code className="bg-white/10 px-2 py-0.5 rounded font-mono text-foreground">{DEMO_EMAIL}</code> / <code className="bg-white/10 px-2 py-0.5 rounded font-mono text-foreground">{DEMO_PASSWORD}</code>
            </p>
          </div>

          <p className="mt-8 text-center text-[10px] font-medium text-muted-foreground uppercase tracking-[0.2em]">
            VisionLock &copy; {year ?? '...'} — Secure neural link
          </p>
        </div>
      </main>
    );
  }

  return (
    <main className="min-h-screen bg-background flex flex-col items-center justify-center p-6 relative overflow-hidden">
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_80%_at_50%_-20%,hsl(var(--primary)/0.08),transparent)]" />
      <div className="absolute inset-0 bg-[linear-gradient(to_right,hsl(var(--border)/0.3)_1px,transparent_1px),linear-gradient(to_bottom,hsl(var(--border)/0.3)_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_70%_60%_at_50%_50%,black,transparent)]" />
      <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-primary/5 rounded-full blur-[120px] pointer-events-none" />
      <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] bg-secondary/5 rounded-full blur-[120px] pointer-events-none" />

      <div className="absolute top-6 right-6 z-20">
        <Button
          variant="outline"
          size="sm"
          onClick={handleLogout}
          className="rounded-xl gap-2 text-[10px] font-bold uppercase tracking-widest border-white/10 bg-white/5 hover:bg-white/10 text-foreground"
        >
          <LogOut className="w-3.5 h-3.5" /> Log out
        </Button>
      </div>

      <div className="relative z-10 w-full max-w-6xl flex flex-col items-center">
        <div className="w-full animate-scale-in">
          <VisionFeed />
        </div>
      </div>

      <footer className="mt-16 text-muted-foreground text-[10px] font-medium tracking-[0.3em] uppercase flex items-center gap-4 relative z-10">
        <span>VisionLock &copy; {year ?? '...'}</span>
        <div className="w-1 h-1 rounded-full bg-primary/60" />
        <span className="text-primary font-bold">PRO.CORE_V3</span>
      </footer>
    </main>
  );
}
