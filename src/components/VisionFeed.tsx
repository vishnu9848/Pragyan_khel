"use client"

import React, { useRef, useState, useEffect, useCallback, useMemo } from 'react';
import { Button } from '@/components/ui/button';
import { Camera, CameraOff, Sparkles, RefreshCw, AlertCircle, Box, Cpu, MousePointer2, XCircle, Moon, Sun, Upload, Video, Maximize, Search, Activity } from 'lucide-react';
import { Card, CardContent, CardFooter, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import { useToast } from '@/hooks/use-toast';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs";

// TensorFlow imports
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

const calculateIoU = (bbox1: [number, number, number, number], bbox2: [number, number, number, number]): number => {
  const [x1, y1, w1, h1] = bbox1;
  const [x2, y2, w2, h2] = bbox2;

  const x_left = Math.max(x1, x2);
  const y_top = Math.max(y1, y2);
  const x_right = Math.min(x1 + w1, x2 + w2);
  const y_bottom = Math.min(y1 + h1, y2 + h2);

  if (x_right < x_left || y_bottom < y_top) {
    return 0.0;
  }

  const intersection_area = (x_right - x_left) * (y_bottom - y_top);
  const area1 = w1 * h1;
  const area2 = w2 * h2;
  const union_area = area1 + area2 - intersection_area;

  return intersection_area / union_area;
};

const lerp = (start: number, end: number, factor: number) => {
  return start + (end - start) * factor;
};

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
  
  // Zoom & Pan interpolation refs
  const zoomFactorRef = useRef(1);
  const panXRef = useRef(0);
  const panYRef = useRef(0);
  
  const [isStreaming, setIsStreaming] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [hasCameraPermission, setHasCameraPermission] = useState<boolean | null>(null);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [isLowLight, setIsLowLight] = useState(false);
  const [isAutoZoomEnabled, setIsAutoZoomEnabled] = useState(true);
  const [isReacquiring, setIsReacquiring] = useState(false);
  const [sourceMode, setSourceMode] = useState<'camera' | 'file'>('camera');
  const [videoFileUrl, setVideoFileUrl] = useState<string | null>(null);
  const [isMounted, setIsMounted] = useState(false);
  
  const [selectedLabel, setSelectedLabel] = useState<string | null>(null);
  const { toast } = useToast();

  useEffect(() => {
    setIsMounted(true);
  }, []);

  useEffect(() => {
    const loadModel = async () => {
      try {
        setIsModelLoading(true);
        await tf.ready();
        const loadedModel = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
        setModel(loadedModel);
      } catch (err) {
        toast({
          title: "AI Core Error",
          description: "Failed to initialize detection engine.",
          variant: "destructive",
        });
      } finally {
        setIsModelLoading(false);
      }
    };
    loadModel();
  }, [toast]);

  const stopStream = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    if (videoFileUrl) {
      URL.revokeObjectURL(videoFileUrl);
      setVideoFileUrl(null);
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.src = "";
    }
    setIsStreaming(false);
    predictionsRef.current = [];
    selectedObjectRef.current = null;
    selectedHistoryRef.current = [];
    trackingConfidenceRef.current = 0;
    reacquisitionCountRef.current = 0;
    setIsReacquiring(false);
    setSelectedLabel(null);
    zoomFactorRef.current = 1;
    
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, [stream, videoFileUrl]);

  const startCamera = async () => {
    setIsLoading(true);
    setHasCameraPermission(null);
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
      toast({
        title: "Access Denied",
        description: "Please enable camera permissions.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
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
        videoRef.current.loop = true;
        await videoRef.current.play();
      }
      setIsStreaming(true);
    } catch (err) {
      toast({
        title: "Import Error",
        description: "Failed to load video file.",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isStreaming || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    let x = (e.clientX - rect.left) * (canvas.width / rect.width);
    let y = (e.clientY - rect.top) * (canvas.height / rect.height);

    if (isAutoZoomEnabled) {
      const zoom = zoomFactorRef.current;
      const panX = panXRef.current;
      const panY = panYRef.current;
      x = (x - canvas.width / 2) / zoom + panX;
      y = (y - canvas.height / 2) / zoom + panY;
    }

    const candidates = predictionsRef.current.filter(prediction => {
      const [bboxX, bboxY, width, height] = prediction.bbox;
      return x >= bboxX && x <= bboxX + width && y >= bboxY && y <= bboxY + height;
    });

    if (candidates.length > 0) {
      const clickedObj = candidates.sort((a, b) => (a.bbox[2] * a.bbox[3]) - (b.bbox[2] * b.bbox[3]))[0];
      selectedObjectRef.current = JSON.parse(JSON.stringify(clickedObj));
      selectedHistoryRef.current = [];
      trackingConfidenceRef.current = 1.0;
      reacquisitionCountRef.current = 0;
      setIsReacquiring(false);
      setSelectedLabel(clickedObj.class);
    } else {
      selectedObjectRef.current = null;
      selectedHistoryRef.current = [];
      trackingConfidenceRef.current = 0;
      reacquisitionCountRef.current = 0;
      setIsReacquiring(false);
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
      const ctx = canvas.getContext('2d', { alpha: false });
      if (!ctx || video.readyState < 2 || video.videoWidth === 0) {
        animationFrameId = requestAnimationFrame(render);
        return;
      }

      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        panXRef.current = canvas.width / 2;
        panYRef.current = canvas.height / 2;
      }

      const detectionFrequency = isReacquiring ? 3 : 10;
      if (frameCountRef.current % detectionFrequency === 0 && model && !isDetectingRef.current) {
        isDetectingRef.current = true;
        model.detect(video).then(predictions => {
          predictionsRef.current = predictions;
          const currentSelected = selectedObjectRef.current;
          
          if (currentSelected) {
            let bestMatch: cocoSsd.DetectedObject | null = null;
            let maxIoU = 0;
            
            for (const prediction of predictions) {
              if (prediction.class === currentSelected.class) {
                const iou = calculateIoU(prediction.bbox, currentSelected.bbox);
                if (iou > maxIoU) { 
                  maxIoU = iou; 
                  bestMatch = prediction; 
                }
              }
            }
            
            if (maxIoU > 0.3 && bestMatch) { 
              selectedObjectRef.current = bestMatch;
              trackingConfidenceRef.current = maxIoU;
              reacquisitionCountRef.current = 0;
              setIsReacquiring(false);
              
              selectedHistoryRef.current.push([...bestMatch.bbox]);
              if (selectedHistoryRef.current.length > 5) {
                selectedHistoryRef.current.shift();
              }
            } else {
              reacquisitionCountRef.current++;
              if (reacquisitionCountRef.current >= 3) {
                setIsReacquiring(true);
                trackingConfidenceRef.current = 0;
                if (reacquisitionCountRef.current > 60) {
                   selectedObjectRef.current = null;
                   selectedHistoryRef.current = [];
                   setIsReacquiring(false);
                   setSelectedLabel(null);
                }
              }
            }
          }
          isDetectingRef.current = false;
        }).catch(() => { isDetectingRef.current = false; });
      }
      frameCountRef.current++;

      const activeSelection = selectedObjectRef.current;
      let targetZoom = 1.0;
      let targetPanX = canvas.width / 2;
      let targetPanY = canvas.height / 2;

      if (activeSelection && isAutoZoomEnabled && !isReacquiring) {
        const [x, y, w, h] = activeSelection.bbox;
        targetZoom = 1.45;
        targetPanX = x + w / 2;
        targetPanY = y + h / 2;
      }

      zoomFactorRef.current = lerp(zoomFactorRef.current, targetZoom, 0.08);
      panXRef.current = lerp(panXRef.current, targetPanX, 0.08);
      panYRef.current = lerp(panYRef.current, targetPanY, 0.08);

      const zoom = zoomFactorRef.current;
      const panX = panXRef.current;
      const panY = panYRef.current;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const enhancementFilter = isLowLight ? "contrast(1.2) brightness(1.1) " : "";

      ctx.save();
      ctx.translate(canvas.width / 2, canvas.height / 2);
      ctx.scale(zoom, zoom);
      ctx.translate(-panX, -panY);
      
      ctx.filter = `${enhancementFilter}blur(16px) brightness(0.5)`;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      if (activeSelection && !isReacquiring) {
        const [x, y, width, height] = activeSelection.bbox;
        ctx.save();
        ctx.beginPath();
        ctx.rect(x, y, width, height);
        ctx.clip();
        ctx.filter = isLowLight ? "contrast(1.2) brightness(1.1)" : "none";
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.restore();
      }

      ctx.filter = "none";

      if (activeSelection && !isReacquiring && selectedHistoryRef.current.length > 1) {
        selectedHistoryRef.current.forEach((trailBbox, index) => {
          if (index === selectedHistoryRef.current.length - 1) return;
          const alpha = (index + 1) / selectedHistoryRef.current.length * 0.3;
          ctx.strokeStyle = `rgba(34, 197, 94, ${alpha})`;
          ctx.lineWidth = 1.5;
          ctx.strokeRect(trailBbox[0], trailBbox[1], trailBbox[2], trailBbox[3]);
        });
      }

      predictionsRef.current.forEach(prediction => {
        const [x, y, width, height] = prediction.bbox;
        const isSelected = activeSelection && prediction.class === activeSelection.class && calculateIoU(prediction.bbox, activeSelection.bbox) > 0.8;
        
        const primaryColor = isSelected ? '#22c55e' : 'rgba(255, 255, 255, 0.2)';
        const labelBg = isSelected ? 'rgba(34, 197, 94, 0.9)' : 'rgba(0, 0, 0, 0.5)';

        ctx.strokeStyle = primaryColor;
        ctx.lineWidth = isSelected ? 4 : 1;
        ctx.setLineDash(isSelected ? [] : [5, 5]);
        ctx.strokeRect(x, y, width, height);
        ctx.setLineDash([]);

        if (isSelected || frameCountRef.current % 30 < 15) {
          let labelText = `${prediction.class}`;
          if (isSelected) {
            const conf = Math.round(trackingConfidenceRef.current * 100);
            labelText = `${prediction.class} • ${conf}%`;
          }
          
          ctx.font = `500 ${Math.max(12, canvas.width * 0.01)}px 'Inter', sans-serif`;
          const textWidth = ctx.measureText(labelText).width;
          ctx.fillStyle = labelBg;
          ctx.fillRect(x, y - (canvas.width * 0.025), textWidth + 12, canvas.width * 0.025);
          ctx.fillStyle = 'white';
          ctx.fillText(labelText, x + 6, y - (canvas.width * 0.008));
        }
      });

      if (isReacquiring && activeSelection) {
        const [x, y, w, h] = activeSelection.bbox;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)';
        ctx.fillRect(x, y + h / 2 - 20, w, 40);
        ctx.fillStyle = '#22c55e';
        ctx.font = "600 14px 'Inter', sans-serif";
        ctx.textAlign = 'center';
        ctx.fillText('REACQUIRING...', x + w / 2, y + h / 2 + 5);
        ctx.textAlign = 'start';
      }

      ctx.restore();
      animationFrameId = requestAnimationFrame(render);
    };

    if (isStreaming) animationFrameId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animationFrameId);
  }, [isStreaming, model, isLowLight, isAutoZoomEnabled, isReacquiring]);

  if (!isMounted) return null;

  return (
    <div className="w-full space-y-8 animate-fade-in">
      <Card className="glass-panel overflow-hidden border-none rounded-[2.5rem]">
        <CardHeader className="text-center pb-8 border-b border-white/5 relative bg-gradient-to-b from-white/[0.02] to-transparent">
          {selectedLabel && (
            <div className="absolute top-8 right-8 animate-scale-in flex flex-col items-end gap-3">
              <Badge className={cn(
                "gap-2 pl-4 py-2 pr-2 shadow-2xl border-none transition-all duration-500 rounded-full",
                isReacquiring ? "bg-amber-500/20 text-amber-500 animate-pulse" : "bg-accent/20 text-accent neon-glow"
              )}>
                {isReacquiring ? <Activity className="w-3 h-3" /> : <Box className="w-3 h-3" />}
                <span className="text-[10px] font-black uppercase tracking-widest">
                  {isReacquiring ? "Searching" : "Locked"}
                </span>
                <span className="font-bold text-sm px-1">{selectedLabel}</span>
                <Button 
                  variant="ghost" 
                  size="icon" 
                  className="h-6 w-6 rounded-full hover:bg-white/10 p-0 ml-1" 
                  onClick={() => { selectedObjectRef.current = null; setSelectedLabel(null); }}
                >
                  <XCircle className="w-4 h-4 opacity-60" />
                </Button>
              </Badge>
            </div>
          )}
          
          <div className="flex justify-center mb-6">
            {isModelLoading ? (
              <Badge variant="outline" className="animate-pulse border-white/10 bg-white/5 gap-2 px-6 py-2 rounded-full text-[10px] uppercase tracking-widest font-bold">
                <RefreshCw className="w-3 h-3 animate-spin" /> Neural Sync Initializing
              </Badge>
            ) : (
              <Badge variant="outline" className="text-accent border-accent/20 gap-2 px-6 py-2 bg-accent/5 rounded-full text-[10px] uppercase tracking-widest font-bold">
                <Activity className="w-3 h-3" /> System Operational
              </Badge>
            )}
          </div>
          
          <CardTitle className="text-4xl md:text-5xl font-black tracking-tighter text-white flex flex-col md:flex-row items-center justify-center gap-6">
            <div className="p-4 bg-primary/20 rounded-3xl border border-primary/20 animate-float">
              <Sparkles className="w-8 h-8 text-primary shadow-primary shadow-2xl" />
            </div>
            Vision Canvas
          </CardTitle>
          <CardDescription className="text-muted-foreground/80 max-w-xl mx-auto mt-6 text-sm font-medium leading-relaxed">
            Next-generation AI tracking with <span className="text-white">Cinematic Auto-Focus</span>. 
            Click any object to lock neural focus and enable depth-of-field isolation.
          </CardDescription>
        </CardHeader>
        
        <CardContent className="p-0 relative bg-black/40">
          <div className={cn(
            "relative aspect-video flex flex-col items-center justify-center overflow-hidden transition-all duration-1000",
            !isStreaming && "opacity-80"
          )}>
            <video ref={videoRef} autoPlay playsInline muted className="absolute inset-0 w-full h-full object-cover opacity-0 pointer-events-none" />
            <canvas ref={canvasRef} onClick={handleCanvasClick} className={cn(
              "absolute inset-0 z-10 transition-opacity duration-1000 w-full h-full object-cover cursor-crosshair",
              isStreaming ? "opacity-100" : "opacity-0 pointer-events-none"
            )} />

            {!isStreaming && !isLoading && (
              <div className="z-10 flex flex-col items-center gap-8 animate-scale-in text-center">
                <div className="p-12 rounded-[3rem] bg-white/5 border border-white/10 backdrop-blur-3xl ring-1 ring-white/10">
                  <Video className="w-16 h-16 text-white/20" />
                </div>
                <div className="space-y-3">
                  <p className="text-xl font-black tracking-tight text-white uppercase italic">Awaiting Input</p>
                  <p className="text-muted-foreground/60 text-xs font-mono tracking-widest">ESTABLISH FEED TO BEGIN ANALYSIS</p>
                </div>
              </div>
            )}

            {isLoading && (
              <div className="absolute inset-0 bg-black/80 z-20 flex flex-col items-center justify-center backdrop-blur-2xl">
                <div className="relative">
                  <RefreshCw className="w-16 h-16 text-primary animate-spin mb-6" />
                  <div className="absolute inset-0 bg-primary/20 blur-2xl" />
                </div>
                <p className="font-black text-xs tracking-[0.3em] text-primary uppercase animate-pulse">Establishing Secure Stream</p>
              </div>
            )}
          </div>
        </CardContent>

        <CardFooter className="flex flex-col gap-10 p-10 bg-white/[0.02]">
          <div className="flex flex-col lg:flex-row items-end justify-between w-full gap-8">
            <div className="flex flex-col gap-5 w-full lg:w-auto">
              <Label className="text-[10px] font-black uppercase tracking-[0.2em] text-muted-foreground ml-1">Input Intelligence</Label>
              <Tabs value={sourceMode} onValueChange={(val) => { stopStream(); setSourceMode(val as any); }} className="w-full sm:w-[320px]">
                <TabsList className="grid w-full grid-cols-2 h-14 bg-black/40 border border-white/5 p-1 rounded-2xl">
                  <TabsTrigger value="camera" className="rounded-xl gap-2 data-[state=active]:bg-primary data-[state=active]:text-white">
                    <Camera className="w-4 h-4" /> Camera
                  </TabsTrigger>
                  <TabsTrigger value="file" className="rounded-xl gap-2 data-[state=active]:bg-primary data-[state=active]:text-white">
                    <Upload className="w-4 h-4" /> Files
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </div>

            <div className="flex flex-wrap items-center gap-5 justify-center lg:justify-end w-full lg:w-auto">
              <div className="flex items-center space-x-8 bg-black/40 px-6 py-4 rounded-2xl border border-white/5">
                <div className="flex items-center gap-4">
                  <div className={cn("p-2 rounded-lg transition-colors", isAutoZoomEnabled ? "bg-accent/10 text-accent" : "bg-white/5 text-muted-foreground")}>
                    <Maximize className="w-4 h-4" />
                  </div>
                  <div className="flex flex-col">
                    <Label htmlFor="auto-zoom" className="text-[10px] font-black uppercase tracking-widest cursor-pointer">Auto-Zoom</Label>
                  </div>
                  <Switch id="auto-zoom" checked={isAutoZoomEnabled} onCheckedChange={setIsAutoZoomEnabled} disabled={!isStreaming} className="data-[state=checked]:bg-accent" />
                </div>
                
                <div className="w-px h-8 bg-white/5" />

                <div className="flex items-center gap-4">
                  <div className={cn("p-2 rounded-lg transition-colors", isLowLight ? "bg-primary/10 text-primary" : "bg-white/5 text-muted-foreground")}>
                    {isLowLight ? <Moon className="w-4 h-4" /> : <Sun className="w-4 h-4" />}
                  </div>
                  <div className="flex flex-col">
                    <Label htmlFor="low-light" className="text-[10px] font-black uppercase tracking-widest cursor-pointer">Enhanced</Label>
                  </div>
                  <Switch id="low-light" checked={isLowLight} onCheckedChange={setIsLowLight} disabled={!isStreaming} className="data-[state=checked]:bg-primary" />
                </div>
              </div>

              {!isStreaming ? (
                sourceMode === 'camera' ? (
                  <Button onClick={startCamera} disabled={isLoading || isModelLoading} className="h-14 px-10 rounded-2xl text-sm font-black uppercase tracking-widest transition-all hover:scale-[1.02] active:scale-95 bg-primary hover:bg-primary/90 shadow-2xl shadow-primary/20">
                    <Camera className="mr-3 h-4 w-4" /> Initialize Feed
                  </Button>
                ) : (
                  <Button onClick={() => fileInputRef.current?.click()} disabled={isLoading || isModelLoading} className="h-14 px-10 rounded-2xl text-sm font-black uppercase tracking-widest transition-all hover:scale-[1.02] active:scale-95 bg-primary hover:bg-primary/90 shadow-2xl shadow-primary/20">
                    <Upload className="mr-3 h-4 w-4" /> Select Media
                  </Button>
                )
              ) : (
                <Button variant="destructive" onClick={stopStream} className="h-14 px-10 rounded-2xl text-sm font-black uppercase tracking-widest transition-all hover:scale-[1.02] active:scale-95 bg-red-500/10 hover:bg-red-500/20 text-red-500 border border-red-500/20">
                  <CameraOff className="mr-3 h-4 w-4" /> Terminate Stream
                </Button>
              )}
            </div>
          </div>

          <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="video/mp4" className="hidden" />

          {hasCameraPermission === false && (
            <Alert className="border-red-500/20 bg-red-500/5 text-red-500 rounded-2xl">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle className="font-black text-[10px] uppercase tracking-widest">Hardware Blocked</AlertTitle>
              <AlertDescription className="text-xs opacity-80">
                Camera access is required for real-time analysis. Please check system permissions.
              </AlertDescription>
            </Alert>
          )}
        </CardFooter>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {[
          { title: "Neural Tracking", desc: "Proprietary IoU algorithms maintain subject lock across dynamic frame shifts.", icon: <Activity className="w-5 h-5" /> },
          { title: "Smart Zoom", desc: "Interpolated viewport scaling (1.5x) ensures jitter-free subject centering.", icon: <Maximize className="w-5 h-5" /> },
          { title: "Bokeh Isolation", desc: "Real-time Gaussian masks provide selective focus on prioritized targets.", icon: <Sparkles className="w-5 h-5" /> }
        ].map((feature, i) => (
          <div key={i} className="glass-panel p-8 rounded-[2rem] space-y-4 hover:bg-white/[0.03] transition-colors group">
            <div className="p-3 bg-primary/10 text-primary w-fit rounded-xl border border-primary/10 group-hover:scale-110 transition-transform">
              {feature.icon}
            </div>
            <h3 className="text-sm font-black uppercase tracking-widest text-white">{feature.title}</h3>
            <p className="text-xs text-muted-foreground/60 leading-relaxed font-medium">
              {feature.desc}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
};