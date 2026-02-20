"use client"

import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Camera, CameraOff, Sparkles, RefreshCw } from 'lucide-react';
import { Card, CardContent, CardFooter, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import { useToast } from '@/hooks/use-toast';

export const VisionFeed: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const { toast } = useToast();

  const startCamera = async () => {
    setIsLoading(true);
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { 
          facingMode: 'user',
          width: { ideal: 1280 },
          height: { ideal: 720 }
        },
        audio: false,
      });
      
      setStream(mediaStream);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
      setIsStreaming(true);
    } catch (err) {
      console.error("Error accessing camera:", err);
      toast({
        title: "Camera Access Failed",
        description: "Please ensure you have granted camera permissions.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
    
    // Clear canvas
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, [stream]);

  // Effect to handle canvas resizing and basic animation overlay
  useEffect(() => {
    let animationFrameId: number;

    const render = () => {
      if (!isStreaming || !canvasRef.current || !videoRef.current) return;
      
      const canvas = canvasRef.current;
      const video = videoRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Sync canvas dimensions with video display size
      if (canvas.width !== video.clientWidth || canvas.height !== video.clientHeight) {
        canvas.width = video.clientWidth;
        canvas.height = video.clientHeight;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Add a subtle sophisticated overlay effect
      const time = Date.now() * 0.001;
      ctx.strokeStyle = 'rgba(208, 188, 255, 0.4)'; // Soft Lavender
      ctx.lineWidth = 2;
      
      // Draw some dynamic corners to simulate a scanner/vision UI
      const size = 30;
      const margin = 20;
      
      // Top Left
      ctx.beginPath();
      ctx.moveTo(margin, margin + size);
      ctx.lineTo(margin, margin);
      ctx.lineTo(margin + size, margin);
      ctx.stroke();

      // Top Right
      ctx.beginPath();
      ctx.moveTo(canvas.width - margin - size, margin);
      ctx.lineTo(canvas.width - margin, margin);
      ctx.lineTo(canvas.width - margin, margin + size);
      ctx.stroke();

      // Bottom Right
      ctx.beginPath();
      ctx.moveTo(canvas.width - margin, canvas.height - margin - size);
      ctx.lineTo(canvas.width - margin, canvas.height - margin);
      ctx.lineTo(canvas.width - margin - size, canvas.height - margin);
      ctx.stroke();

      // Bottom Left
      ctx.beginPath();
      ctx.moveTo(margin + size, canvas.height - margin);
      ctx.lineTo(margin, canvas.height - margin);
      ctx.lineTo(margin, canvas.height - margin - size);
      ctx.stroke();

      // Center crosshair with a pulse
      const pulse = Math.sin(time * 3) * 5;
      ctx.beginPath();
      ctx.arc(canvas.width / 2, canvas.height / 2, 5 + pulse, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(103, 80, 164, 0.5)'; // Deep Purple
      ctx.fill();

      animationFrameId = requestAnimationFrame(render);
    };

    if (isStreaming) {
      animationFrameId = requestAnimationFrame(render);
    }

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [isStreaming]);

  return (
    <div className="w-full max-w-4xl mx-auto p-4 md:p-8 space-y-8 animate-fade-in">
      <Card className="overflow-hidden shadow-2xl border-none bg-white/80 backdrop-blur-sm">
        <CardHeader className="text-center pb-2">
          <CardTitle className="text-4xl font-headline font-bold text-primary flex items-center justify-center gap-3">
            <Sparkles className="w-8 h-8" />
            Vision Canvas
          </CardTitle>
          <CardDescription className="text-lg">
            Real-time visual intelligence and canvas interaction.
          </CardDescription>
        </CardHeader>
        
        <CardContent className="p-0 relative group">
          <div className={cn(
            "relative aspect-video bg-muted flex items-center justify-center overflow-hidden transition-all duration-500",
            !isStreaming && "bg-slate-100"
          )}>
            {!isStreaming && (
              <div className="flex flex-col items-center gap-4 text-muted-foreground animate-scale-in">
                <div className="p-8 rounded-full bg-secondary/50">
                  <Camera className="w-16 h-16 text-primary" />
                </div>
                <p className="font-medium">Camera feed is currently offline</p>
              </div>
            )}
            
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={cn(
                "w-full h-full object-cover transition-opacity duration-700",
                isStreaming ? "opacity-100" : "opacity-0"
              )}
            />
            
            <canvas
              ref={canvasRef}
              className={cn(
                "absolute inset-0 pointer-events-none transition-opacity duration-700",
                isStreaming ? "opacity-100" : "opacity-0"
              )}
            />

            {isLoading && (
              <div className="absolute inset-0 bg-white/40 flex items-center justify-center backdrop-blur-[2px]">
                <RefreshCw className="w-12 h-12 text-primary animate-spin" />
              </div>
            )}
          </div>
        </CardContent>

        <CardFooter className="flex flex-col sm:flex-row gap-4 p-8 justify-center bg-secondary/10">
          {!isStreaming ? (
            <Button 
              size="lg" 
              onClick={startCamera} 
              disabled={isLoading}
              className="px-10 h-14 rounded-full text-lg font-semibold transition-all hover:scale-105 active:scale-95 shadow-lg shadow-primary/20"
            >
              {isLoading ? (
                <><RefreshCw className="mr-2 h-5 w-5 animate-spin" /> Starting...</>
              ) : (
                <><Camera className="mr-2 h-5 w-5" /> Start Camera</>
              )}
            </Button>
          ) : (
            <Button 
              variant="destructive" 
              size="lg" 
              onClick={stopCamera}
              className="px-10 h-14 rounded-full text-lg font-semibold transition-all hover:scale-105 active:scale-95 shadow-lg shadow-destructive/20"
            >
              <CameraOff className="mr-2 h-5 w-5" /> Stop Stream
            </Button>
          )}
          
          <div className="hidden sm:flex items-center gap-2 px-6 py-3 rounded-full bg-white/50 border border-border text-sm font-medium text-muted-foreground">
            <div className={cn(
              "w-2.5 h-2.5 rounded-full",
              isStreaming ? "bg-green-500 animate-pulse" : "bg-slate-300"
            )} />
            {isStreaming ? "Live Feed Active" : "Waiting for Access"}
          </div>
        </CardFooter>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 animate-fade-in delay-200">
        {[
          { title: "Low Latency", desc: "Native browser stream ensures real-time performance.", icon: "⚡" },
          { title: "Layered UI", desc: "Interactive canvas overlay perfectly synced with video.", icon: "🎨" },
          { title: "Privacy First", desc: "No data is sent to servers. Processing happens on-device.", icon: "🔒" }
        ].map((feature, i) => (
          <Card key={i} className="bg-white/60 border-none shadow-sm hover:shadow-md transition-shadow">
            <CardHeader className="pb-2">
              <div className="text-2xl mb-2">{feature.icon}</div>
              <CardTitle className="text-lg font-headline">{feature.title}</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground leading-relaxed">
                {feature.desc}
              </p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
};