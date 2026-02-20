"use client"

import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Camera, CameraOff, RefreshCw, AlertCircle, Box, Cpu, XCircle, Moon, Sun, Upload, Video, Maximize, Activity, Gauge } from 'lucide-react';
import { Card, CardContent, CardFooter, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import { useToast } from '@/hooks/use-toast';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

const calculateIoU = (bbox1: [number, number, number, number], bbox2: [number, number, number, number]): number => {
  const [x1, y1, w1, h1] = bbox1;
  const [x2, y2, w2, h2] = bbox2;
  const x_left = Math.max(x1, x2);
  const y_top = Math.max(y1, y2);
  const x_right = Math.min(x1 + w1, x2 + w2);
  const y_bottom = Math.min(y1 + h1, y2 + h2);
  if (x_right < x_left || y_bottom < y_top) return 0.0;
  const intersection_area = (x_right - x_left) * (y_bottom - y_top);
  const union_area = (w1 * h1) + (w2 * h2) - intersection_area;
  return intersection_area / union_area;
};

const lerp = (start: number, end: number, factor: number) => start + (end - start) * factor;

export const VisionFeed: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const predictionsRef = useRef<cocoSsd.DetectedObject[]>([]);
  const frameCountRef = useRef(0);
  const isDetectingRef = useRef(false);
  const selectedObjectRef = useRef<cocoSsd.DetectedObject | null>(null);
  const selectedHistoryRef = useRef<[number, number, number, number][]>([]);
  const trackingConfidenceRef = useRef<number>(0);
  const reacquisitionCountRef = useRef(0);
  
  const lastFpsUpdateRef = useRef(performance.now());
  const frameCountSinceUpdateRef = useRef(0);
  const detectionTimeRef = useRef(0);
  
  const zoomFactorRef = useRef(1);
  const panXRef = useRef(0);
  const panYRef = useRef(0);
  const focusBboxRef = useRef<[number, number, number, number] | null>(null);
  const focusAlphaRef = useRef(0);
  const focusScaleRef = useRef(0.8);
  
  const [isStreaming, setIsStreaming] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [hasCameraPermission, setHasCameraPermission] = useState<boolean | null>(null);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [isLowLight, setIsLowLight] = useState(false);
  const [isAutoZoomEnabled, setIsAutoZoomEnabled] = useState(false); 
  const [isReacquiring, setIsReacquiring] = useState(false);
  const [sourceMode, setSourceMode] = useState<'camera' | 'file'>('camera');
  const [videoFileUrl, setVideoFileUrl] = useState<string | null>(null);
  const [isMounted, setIsMounted] = useState(false);
  
  const [fps, setFps] = useState(0);
  const [inferenceTime, setInferenceTime] = useState(0);
  const [selectedLabel, setSelectedLabel] = useState<string | null>(null);
  const { toast } = useToast();

  useEffect(() => { setIsMounted(true); }, []);

  useEffect(() => {
    const loadModel = async () => {
      try {
        setIsModelLoading(true);
        await tf.ready();
        const loadedModel = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
        setModel(loadedModel);
      } catch (err) {
        toast({ title: "Inference Error", description: "Could not load AI vision core.", variant: "destructive" });
      } finally { setIsModelLoading(false); }
    };
    loadModel();
  }, [toast]);

  const stopStream = useCallback(() => {
    if (stream) stream.getTracks().forEach(track => track.stop());
    if (videoFileUrl) URL.revokeObjectURL(videoFileUrl);
    setStream(null);
    setVideoFileUrl(null);
    setIsStreaming(false);
    predictionsRef.current = [];
    selectedObjectRef.current = null;
    setSelectedLabel(null);
    zoomFactorRef.current = 1;
    focusAlphaRef.current = 0;
  }, [stream, videoFileUrl]);

  const startCamera = async () => {
    setIsLoading(true);
    try {
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      setStream(mediaStream);
      setHasCameraPermission(true);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        await videoRef.current.play();
      }
      setIsStreaming(true);
    } catch (err) {
      setHasCameraPermission(false);
      toast({ title: "Access Denied", description: "Camera permission is required.", variant: "destructive" });
    } finally { setIsLoading(false); }
  };

  const handleFileChange = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setIsLoading(true);
    try {
      const url = URL.createObjectURL(file);
      setVideoFileUrl(url);
      if (videoRef.current) {
        videoRef.current.src = url;
        await videoRef.current.play();
      }
      setIsStreaming(true);
    } catch (err) {
      toast({ title: "Load Error", description: "Failed to load video file.", variant: "destructive" });
    } finally { setIsLoading(false); }
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isStreaming || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    let x = (e.clientX - rect.left) * (canvas.width / rect.width);
    let y = (e.clientY - rect.top) * (canvas.height / rect.height);

    if (isAutoZoomEnabled && zoomFactorRef.current > 1.05) {
      x = (x - canvas.width / 2) / zoomFactorRef.current + panXRef.current;
      y = (y - canvas.height / 2) / zoomFactorRef.current + panYRef.current;
    }

    const clicked = predictionsRef.current.find(p => {
      const [bx, by, bw, bh] = p.bbox;
      return x >= bx && x <= bx + bw && y >= by && y <= by + bh;
    });

    if (clicked) {
      selectedObjectRef.current = JSON.parse(JSON.stringify(clicked));
      setSelectedLabel(clicked.class);
      focusScaleRef.current = 0.6;
    } else {
      selectedObjectRef.current = null;
      setSelectedLabel(null);
    }
  };

  useEffect(() => {
    let animationFrameId: number;
    const render = () => {
      if (!isStreaming || !canvasRef.current || !videoRef.current) {
        if (isStreaming) animationFrameId = requestAnimationFrame(render);
        return;
      }
      const canvas = canvasRef.current;
      const video = videoRef.current;
      const ctx = canvas.getContext('2d');
      if (!ctx || video.readyState < 2) {
        animationFrameId = requestAnimationFrame(render);
        return;
      }

      if (canvas.width !== video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        panXRef.current = canvas.width / 2;
        panYRef.current = canvas.height / 2;
      }

      const now = performance.now();
      frameCountSinceUpdateRef.current++;
      if (now - lastFpsUpdateRef.current >= 1000) {
        setFps(Math.round((frameCountSinceUpdateRef.current * 1000) / (now - lastFpsUpdateRef.current)));
        setInferenceTime(detectionTimeRef.current);
        lastFpsUpdateRef.current = now;
        frameCountSinceUpdateRef.current = 0;
      }

      if (frameCountRef.current % (isReacquiring ? 4 : 10) === 0 && model && !isDetectingRef.current) {
        isDetectingRef.current = true;
        const start = performance.now();
        model.detect(video).then(predictions => {
          detectionTimeRef.current = Math.round(performance.now() - start);
          predictionsRef.current = predictions;
          const current = selectedObjectRef.current;
          if (current) {
            const best = predictions.filter(p => p.class === current.class)
              .sort((a, b) => calculateIoU(b.bbox, current.bbox) - calculateIoU(a.bbox, current.bbox))[0];
            
            if (best && calculateIoU(best.bbox, current.bbox) > 0.3) {
              selectedObjectRef.current = best;
              trackingConfidenceRef.current = calculateIoU(best.bbox, current.bbox);
              setIsReacquiring(false);
              reacquisitionCountRef.current = 0;
            } else {
              reacquisitionCountRef.current++;
              if (reacquisitionCountRef.current > 5) setIsReacquiring(true);
              if (reacquisitionCountRef.current > 100) { selectedObjectRef.current = null; setSelectedLabel(null); }
            }
          }
          isDetectingRef.current = false;
        }).catch(() => isDetectingRef.current = false);
      }
      frameCountRef.current++;

      const active = selectedObjectRef.current;
      let targetZoom = 1.0;
      let targetPanX = canvas.width / 2;
      let targetPanY = canvas.height / 2;

      if (active && !isReacquiring) {
        const [bx, by, bw, bh] = active.bbox;
        if (!focusBboxRef.current) focusBboxRef.current = [...active.bbox];
        else {
          focusBboxRef.current[0] = lerp(focusBboxRef.current[0], bx, 0.15);
          focusBboxRef.current[1] = lerp(focusBboxRef.current[1], by, 0.15);
          focusBboxRef.current[2] = lerp(focusBboxRef.current[2], bw, 0.15);
          focusBboxRef.current[3] = lerp(focusBboxRef.current[3], bh, 0.15);
        }
        focusAlphaRef.current = lerp(focusAlphaRef.current, 1.0, 0.1);
        focusScaleRef.current = lerp(focusScaleRef.current, 1.0, 0.1);
        if (isAutoZoomEnabled) {
          targetZoom = 1.6;
          targetPanX = focusBboxRef.current[0] + focusBboxRef.current[2] / 2;
          targetPanY = focusBboxRef.current[1] + focusBboxRef.current[3] / 2;
        }
      } else {
        focusAlphaRef.current = lerp(focusAlphaRef.current, 0, 0.1);
        focusScaleRef.current = lerp(focusScaleRef.current, 0.8, 0.1);
      }

      zoomFactorRef.current = lerp(zoomFactorRef.current, targetZoom, 0.08);
      panXRef.current = lerp(panXRef.current, targetPanX, 0.08);
      panYRef.current = lerp(panYRef.current, targetPanY, 0.08);

      ctx.save();
      ctx.translate(canvas.width / 2, canvas.height / 2);
      ctx.scale(zoomFactorRef.current, zoomFactorRef.current);
      ctx.translate(-panXRef.current, -panYRef.current);

      const blur = focusAlphaRef.current * 10;
      const darken = 1.0 - (focusAlphaRef.current * 0.2);
      ctx.filter = `blur(${blur}px) brightness(${darken})`;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      ctx.filter = "none";

      if (focusBboxRef.current && focusAlphaRef.current > 0.01) {
        const [fx, fy, fw, fh] = focusBboxRef.current;
        const s = focusScaleRef.current;
        const dw = fw * s, dh = fh * s;
        const dx = fx + (fw - dw) / 2, dy = fy + (fh - dh) / 2;
        ctx.save();
        ctx.globalAlpha = focusAlphaRef.current;
        ctx.beginPath();
        ctx.rect(dx, dy, dw, dh);
        ctx.clip();
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.restore();
        
        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 4;
        ctx.strokeRect(dx, dy, dw, dh);
      }

      predictionsRef.current.forEach(p => {
        if (active && p.class === active.class) return;
        const [px, py, pw, ph] = p.bbox;
        ctx.strokeStyle = 'rgba(16, 185, 129, 0.3)';
        ctx.setLineDash([5, 5]);
        ctx.strokeRect(px, py, pw, ph);
        ctx.setLineDash([]);
      });

      ctx.restore();
      animationFrameId = requestAnimationFrame(render);
    };
    if (isStreaming) animationFrameId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animationFrameId);
  }, [isStreaming, model, isLowLight, isAutoZoomEnabled, isReacquiring]);

  if (!isMounted) return null;

  return (
    <div className="w-full space-y-8 animate-fade-in">
      <Card className="bg-white border-slate-100 shadow-[0_10px_25px_rgba(0,0,0,0.05)] overflow-hidden rounded-[2rem]">
        <CardHeader className="text-center py-10 border-b border-slate-50 bg-slate-50/30 relative">
          {selectedLabel && (
            <div className="absolute top-8 right-8 animate-scale-in z-30">
              <Badge className={cn(
                "gap-3 pl-4 py-2 pr-2 shadow-sm border-none transition-all rounded-full font-bold",
                isReacquiring ? "bg-amber-100 text-amber-600 animate-pulse" : "bg-primary/10 text-primary emerald-glow"
              )}>
                <span className="text-[10px] uppercase tracking-widest">{isReacquiring ? "Seeking" : "Locked"}</span>
                <span className="text-sm px-1">{selectedLabel}</span>
                <Button variant="ghost" size="icon" className="h-6 w-6 rounded-full hover:bg-black/5" onClick={() => { selectedObjectRef.current = null; setSelectedLabel(null); }}>
                  <XCircle className="w-4 h-4 opacity-40" />
                </Button>
              </Badge>
            </div>
          )}
          
          <div className="flex justify-center mb-6">
            <Badge variant="outline" className={cn(
              "gap-2.5 px-6 py-1.5 rounded-full text-[10px] uppercase tracking-widest font-black",
              isModelLoading ? "text-slate-400 border-slate-200" : "text-primary border-primary/20 bg-primary/5"
            )}>
              {isModelLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Cpu className="w-3.5 h-3.5" />}
              {isModelLoading ? "Initializing Core" : "Neural Link Active"}
            </Badge>
          </div>
          
          <CardTitle className="text-4xl md:text-5xl font-black tracking-tight text-slate-900">
            Vision Canvas
          </CardTitle>
          <CardDescription className="text-slate-500 max-w-lg mx-auto mt-4 text-sm font-medium leading-relaxed">
            Premium spatial intelligence with <span className="text-primary font-bold">Dynamic Focus</span>.
            Select any object to isolate it from the environment.
          </CardDescription>
        </CardHeader>
        
        <CardContent className="p-0 relative bg-slate-100/20">
          <div className={cn("relative aspect-video flex items-center justify-center overflow-hidden", !isStreaming && "opacity-40")}>
            <video ref={videoRef} autoPlay playsInline muted className="hidden" />
            <canvas ref={canvasRef} onClick={handleCanvasClick} className={cn(
              "absolute inset-0 z-10 w-full h-full object-cover cursor-crosshair transition-opacity duration-1000",
              isStreaming ? "opacity-100" : "opacity-0"
            )} />

            {!isStreaming && !isLoading && (
              <div className="z-10 flex flex-col items-center gap-8 animate-scale-in text-center p-12">
                <div className="p-12 rounded-[3rem] bg-white border border-slate-200 shadow-sm animate-float">
                   <div className="absolute inset-0 bg-primary/5 blur-3xl rounded-full scale-75 animate-pulse" />
                   <Video className="w-16 h-16 text-slate-200 relative z-10" />
                </div>
                <div className="space-y-3 relative z-10">
                  <div className="flex items-center justify-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-primary animate-ping" />
                    <p className="text-xl font-bold tracking-widest text-slate-900 uppercase italic">Awaiting Interface</p>
                  </div>
                  <p className="text-slate-400 text-[9px] font-mono tracking-[0.4em] uppercase">Connect Camera to Scan</p>
                </div>
              </div>
            )}

            {isLoading && (
              <div className="absolute inset-0 bg-white/80 z-20 flex flex-col items-center justify-center backdrop-blur-sm">
                <RefreshCw className="w-12 h-12 text-primary animate-spin mb-4" />
                <p className="font-bold text-[10px] tracking-[0.4em] text-slate-900 uppercase">Linking Hardware</p>
              </div>
            )}
            
            {isStreaming && (
              <div className="absolute bottom-6 left-6 z-30 flex gap-2">
                <Badge className="bg-white/90 backdrop-blur-md border border-slate-200 text-slate-900 text-[9px] font-mono py-1 px-3 rounded-full flex gap-2 items-center">
                  <Gauge className="w-3 h-3 text-primary" /> {fps} FPS
                </Badge>
                <Badge className="bg-white/90 backdrop-blur-md border border-slate-200 text-slate-900 text-[9px] font-mono py-1 px-3 rounded-full flex gap-2 items-center">
                  <Cpu className="w-3 h-3 text-primary" /> {inferenceTime}ms INF
                </Badge>
              </div>
            )}
          </div>
        </CardContent>

        <CardFooter className="flex flex-col gap-10 p-10 bg-slate-50/50">
          <div className="flex flex-col lg:flex-row items-center justify-between w-full gap-8">
            <div className="flex flex-col gap-4 w-full lg:w-auto">
              <Label className="text-[9px] font-black uppercase tracking-[0.3em] text-slate-400 ml-1">Input Source</Label>
              <Tabs value={sourceMode} onValueChange={(v) => { stopStream(); setSourceMode(v as any); }} className="w-full sm:w-[320px]">
                <TabsList className="grid w-full grid-cols-2 h-12 bg-white border border-slate-200 p-1 rounded-xl shadow-sm">
                  <TabsTrigger value="camera" className="rounded-lg gap-2 text-[10px] font-bold uppercase tracking-widest data-[state=active]:bg-primary data-[state=active]:text-white transition-all">
                    <Camera className="w-3.5 h-3.5" /> Camera
                  </TabsTrigger>
                  <TabsTrigger value="file" className="rounded-lg gap-2 text-[10px] font-bold uppercase tracking-widest data-[state=active]:bg-primary data-[state=active]:text-white transition-all">
                    <Upload className="w-3.5 h-3.5" /> File
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </div>

            <div className="flex flex-wrap items-center gap-6 justify-center lg:justify-end w-full lg:w-auto">
              <div className="flex items-center space-x-8 bg-white px-8 py-4 rounded-2xl border border-slate-200 shadow-sm">
                <div className="flex flex-col gap-2">
                  <Label htmlFor="auto-zoom" className="text-[9px] font-black uppercase tracking-widest text-slate-500">Cinematic Zoom</Label>
                  <Switch id="auto-zoom" checked={isAutoZoomEnabled} onCheckedChange={setIsAutoZoomEnabled} disabled={!isStreaming} className="data-[state=checked]:bg-primary" />
                </div>
                <div className="w-px h-8 bg-slate-100" />
                <div className="flex flex-col gap-2">
                  <Label htmlFor="low-light" className="text-[9px] font-black uppercase tracking-widest text-slate-500">Enhanced Feed</Label>
                  <Switch id="low-light" checked={isLowLight} onCheckedChange={setIsLowLight} disabled={!isStreaming} className="data-[state=checked]:bg-primary" />
                </div>
              </div>

              {!isStreaming ? (
                <Button onClick={sourceMode === 'camera' ? startCamera : () => fileInputRef.current?.click()} disabled={isLoading || isModelLoading} className="h-14 px-10 rounded-xl text-[10px] font-black uppercase tracking-[0.2em] transition-all hover:scale-[1.03] active:scale-95 bg-primary hover:bg-primary/90 text-white shadow-lg shadow-primary/20">
                  {sourceMode === 'camera' ? <Camera className="mr-3 h-4 w-4" /> : <Upload className="mr-3 h-4 w-4" />}
                  {sourceMode === 'camera' ? "Start Feed" : "Choose File"}
                </Button>
              ) : (
                <Button variant="destructive" onClick={stopStream} className="h-14 px-10 rounded-xl text-[10px] font-black uppercase tracking-[0.2em] transition-all hover:scale-[1.03] active:scale-95 bg-slate-900 hover:bg-slate-800 text-white border-none">
                  <CameraOff className="mr-3 h-4 w-4" /> Kill Link
                </Button>
              )}
            </div>
          </div>
          <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="video/mp4" className="hidden" />
          {hasCameraPermission === false && (
            <Alert variant="destructive" className="bg-red-50 border-red-100 text-red-600 rounded-2xl p-4">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle className="font-bold text-[10px] uppercase tracking-widest mb-1">Access Denied</AlertTitle>
              <AlertDescription className="text-xs">Camera access is blocked by security policy.</AlertDescription>
            </Alert>
          )}
        </CardFooter>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {[
          { title: "Smart Lock", desc: "Maintains object persistence even across frame boundaries and occlusions.", icon: <Activity className="w-5 h-5" /> },
          { title: "Dynamic Depth", desc: "Gaussian background isolation provides high-end cinematic focus.", icon: <Maximize className="w-5 h-5" /> },
          { title: "Neural Logic", desc: "Real-time inference using optimized MobileNet v2 detection matrix.", icon: <Cpu className="w-5 h-5" /> }
        ].map((f, i) => (
          <div key={i} className="bg-white p-8 rounded-2xl border border-slate-100 shadow-sm space-y-4 hover:shadow-md transition-all group">
            <div className="p-3 bg-primary/10 text-primary w-fit rounded-lg border border-primary/5 group-hover:bg-primary group-hover:text-white transition-colors">
              {f.icon}
            </div>
            <h3 className="text-sm font-bold uppercase tracking-widest text-slate-900">{f.title}</h3>
            <p className="text-xs text-slate-500 leading-relaxed font-medium">{f.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
};
