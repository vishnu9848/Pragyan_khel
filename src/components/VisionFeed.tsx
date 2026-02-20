"use client"

import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Camera, CameraOff, Sparkles, RefreshCw, AlertCircle, Box, Cpu, XCircle, Moon, Sun, Upload, Video, Maximize, Activity, Gauge } from 'lucide-react';
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
  
  // Performance Monitor Refs
  const lastFpsUpdateRef = useRef(performance.now());
  const frameCountSinceUpdateRef = useRef(0);
  const detectionTimeRef = useRef(0);
  
  // Animation & Interpolation refs
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
  
  // Performance State
  const [fps, setFps] = useState(0);
  const [inferenceTime, setInferenceTime] = useState(0);
  
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
          title: "Inference Engine Error",
          description: "Could not initialize neural detection core.",
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
    focusBboxRef.current = null;
    focusAlphaRef.current = 0;
    focusScaleRef.current = 0.8;
    setFps(0);
    setInferenceTime(0);
    
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
        title: "Permission Required",
        description: "Please allow camera access to start scanning.",
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
        title: "Load Failure",
        description: "Unsupported video format or corrupted file.",
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

    if (isAutoZoomEnabled && zoomFactorRef.current > 1.05) {
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
      focusScaleRef.current = 0.6; 
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

      const now = performance.now();
      frameCountSinceUpdateRef.current++;
      if (now - lastFpsUpdateRef.current >= 1000) {
        setFps(Math.round((frameCountSinceUpdateRef.current * 1000) / (now - lastFpsUpdateRef.current)));
        setInferenceTime(detectionTimeRef.current);
        lastFpsUpdateRef.current = now;
        frameCountSinceUpdateRef.current = 0;
      }

      const detectionFrequency = isReacquiring ? 3 : 10;
      if (frameCountRef.current % detectionFrequency === 0 && model && !isDetectingRef.current) {
        isDetectingRef.current = true;
        const startDetect = performance.now();
        model.detect(video).then(predictions => {
          detectionTimeRef.current = Math.round(performance.now() - startDetect);
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

      if (activeSelection && !isReacquiring) {
        const [tx, ty, tw, th] = activeSelection.bbox;
        if (!focusBboxRef.current) {
          focusBboxRef.current = [...activeSelection.bbox];
        } else {
          focusBboxRef.current[0] = lerp(focusBboxRef.current[0], tx, 0.15);
          focusBboxRef.current[1] = lerp(focusBboxRef.current[1], ty, 0.15);
          focusBboxRef.current[2] = lerp(focusBboxRef.current[2], tw, 0.15);
          focusBboxRef.current[3] = lerp(focusBboxRef.current[3], th, 0.15);
        }
        focusAlphaRef.current = lerp(focusAlphaRef.current, 1.0, 0.1);
        focusScaleRef.current = lerp(focusScaleRef.current, 1.0, 0.1);
        
        if (isAutoZoomEnabled) {
          targetZoom = 1.8; // More aggressive zoom for cinematic feel
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

      const zoom = zoomFactorRef.current;
      const panX = panXRef.current;
      const panY = panYRef.current;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const enhancementFilter = isLowLight ? "contrast(1.2) brightness(1.1) " : "";
      
      const blurAmount = focusAlphaRef.current * 16;
      const brightnessAmount = 1.0 - (focusAlphaRef.current * 0.4);

      // Start transformation context
      ctx.save();
      ctx.translate(canvas.width / 2, canvas.height / 2);
      ctx.scale(zoom, zoom);
      ctx.translate(-panX, -panY);
      
      // Draw background with blur based on selection state
      ctx.filter = `${enhancementFilter}blur(${blurAmount}px) brightness(${brightnessAmount})`;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      
      // Reset filter for overlays
      ctx.filter = "none";

      // Draw sharp focus area if selection is active
      if (focusBboxRef.current && focusAlphaRef.current > 0.01) {
        const [x, y, w, h] = focusBboxRef.current;
        const scale = focusScaleRef.current;
        const dw = w * scale;
        const dh = h * scale;
        const dx = x + (w - dw) / 2;
        const dy = y + (h - dh) / 2;

        ctx.save();
        ctx.globalAlpha = focusAlphaRef.current;
        ctx.beginPath();
        ctx.rect(dx, dy, dw, dh);
        ctx.clip();
        ctx.filter = isLowLight ? "contrast(1.2) brightness(1.1)" : "none";
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.restore();
      }

      // Draw trails
      if (activeSelection && !isReacquiring && selectedHistoryRef.current.length > 1) {
        selectedHistoryRef.current.forEach((trailBbox, index) => {
          if (index === selectedHistoryRef.current.length - 1) return;
          const alpha = (index + 1) / selectedHistoryRef.current.length * 0.4;
          ctx.strokeStyle = `rgba(170, 255, 230, ${alpha})`;
          ctx.lineWidth = 1.5;
          ctx.strokeRect(trailBbox[0], trailBbox[1], trailBbox[2], trailBbox[3]);
        });
      }

      // Draw detections
      predictionsRef.current.forEach(prediction => {
        const [x, y, width, height] = prediction.bbox;
        const isSelected = activeSelection && prediction.class === activeSelection.class && calculateIoU(prediction.bbox, activeSelection.bbox) > 0.8;
        
        const accentColor = isSelected ? '#14ffd2' : 'rgba(255, 255, 255, 0.2)';
        const labelBg = isSelected ? 'rgba(20, 255, 210, 0.9)' : 'rgba(30, 40, 60, 0.7)';
        const labelTextCol = isSelected ? '#001a14' : '#ffffff';

        ctx.strokeStyle = accentColor;
        ctx.lineWidth = isSelected ? 4 : 1;
        ctx.setLineDash(isSelected ? [] : [4, 4]);
        
        if (isSelected && focusBboxRef.current) {
          const [fx, fy, fw, fh] = focusBboxRef.current;
          ctx.strokeRect(fx, fy, fw, fh);
          
          let labelText = `${prediction.class.toUpperCase()} • ${Math.round(trackingConfidenceRef.current * 100)}%`;
          ctx.font = `700 ${Math.max(12, canvas.width * 0.012)}px 'Inter', sans-serif`;
          const textWidth = ctx.measureText(labelText).width;
          const textHeight = canvas.width * 0.022;
          
          ctx.fillStyle = labelBg;
          ctx.fillRect(fx, fy - textHeight, textWidth + 14, textHeight);
          ctx.fillStyle = labelTextCol;
          ctx.fillText(labelText, fx + 7, fy - (textHeight * 0.35));
        } else if (!activeSelection) {
          ctx.strokeRect(x, y, width, height);
          if (frameCountRef.current % 30 < 15) {
            ctx.font = `700 ${Math.max(12, canvas.width * 0.012)}px 'Inter', sans-serif`;
            const textWidth = ctx.measureText(prediction.class).width;
            const textHeight = canvas.width * 0.022;
            ctx.fillStyle = labelBg;
            ctx.fillRect(x, y - textHeight, textWidth + 14, textHeight);
            ctx.fillStyle = labelTextCol;
            ctx.fillText(prediction.class, x + 7, y - (textHeight * 0.35));
          }
        }
        ctx.setLineDash([]);
      });

      if (isReacquiring && activeSelection) {
        const [x, y, w, h] = activeSelection.bbox;
        ctx.fillStyle = 'rgba(20, 30, 50, 0.85)';
        ctx.fillRect(x, y + h / 2 - 20, w, 40);
        ctx.fillStyle = '#14ffd2';
        ctx.font = "800 12px 'Inter', sans-serif";
        ctx.textAlign = 'center';
        ctx.fillText('REACQUIRING TARGET...', x + w / 2, y + h / 2 + 5);
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
    <div className="w-full space-y-10 animate-fade-in">
      <Card className="glass-panel overflow-hidden border-none rounded-[3rem]">
        <CardHeader className="text-center pb-10 border-b border-white/5 relative bg-gradient-to-b from-white/[0.03] to-transparent">
          {selectedLabel && (
            <div className="absolute top-10 right-10 animate-scale-in flex flex-col items-end gap-3 z-30">
              <Badge className={cn(
                "gap-3 pl-5 py-3 pr-3 shadow-2xl border-none transition-all duration-500 rounded-full",
                isReacquiring ? "bg-amber-500/20 text-amber-500 animate-pulse" : "bg-accent/20 text-accent neon-glow"
              )}>
                {isReacquiring ? <Activity className="w-4 h-4" /> : <Box className="w-4 h-4" />}
                <span className="text-[10px] font-black uppercase tracking-[0.2em]">
                  {isReacquiring ? "Acquiring" : "Locked"}
                </span>
                <span className="font-bold text-base px-1 tracking-tight">{selectedLabel}</span>
                <Button 
                  variant="ghost" 
                  size="icon" 
                  className="h-8 w-8 rounded-full hover:bg-white/10 p-0 ml-1" 
                  onClick={() => { selectedObjectRef.current = null; setSelectedLabel(null); }}
                >
                  <XCircle className="w-5 h-5 opacity-60" />
                </Button>
              </Badge>
            </div>
          )}
          
          <div className="flex justify-center mb-8">
            {isModelLoading ? (
              <Badge variant="outline" className="animate-pulse border-primary/20 bg-primary/5 gap-3 px-8 py-2.5 rounded-full text-[10px] uppercase tracking-[0.25em] font-black text-primary">
                <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Initializing Neural Core
              </Badge>
            ) : (
              <Badge variant="outline" className="text-accent border-accent/20 gap-3 px-8 py-2.5 bg-accent/5 rounded-full text-[10px] uppercase tracking-[0.25em] font-black">
                <Cpu className="w-3.5 h-3.5" /> Neural Engine Active
              </Badge>
            )}
          </div>
          
          <CardTitle className="text-5xl md:text-7xl font-black tracking-tighter text-white flex flex-col items-center justify-center gap-6">
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-white via-white/80 to-white/60">Vision Canvas</span>
          </CardTitle>
          <CardDescription className="text-muted-foreground/60 max-w-2xl mx-auto mt-8 text-base font-medium leading-relaxed tracking-wide">
            Next-gen spatial intelligence with <span className="text-accent font-bold">Cinematic Auto-Focus</span>. 
            Isolate targets with depth-of-field neural masks.
          </CardDescription>
        </CardHeader>
        
        <CardContent className="p-0 relative bg-background/20">
          <div className={cn(
            "relative aspect-video flex flex-col items-center justify-center overflow-hidden transition-all duration-1000",
            !isStreaming && "opacity-60"
          )}>
            <video ref={videoRef} autoPlay playsInline muted className="absolute inset-0 w-full h-full object-cover opacity-0 pointer-events-none" />
            <canvas ref={canvasRef} onClick={handleCanvasClick} className={cn(
              "absolute inset-0 z-10 transition-opacity duration-1000 w-full h-full object-cover cursor-crosshair",
              isStreaming ? "opacity-100" : "opacity-0 pointer-events-none"
            )} />

            {!isStreaming && !isLoading && (
              <div className="z-10 flex flex-col items-center gap-10 animate-scale-in text-center relative w-full h-full">
                {/* Dynamic Background Scanning Animation */}
                <div className="absolute inset-0 pointer-events-none overflow-hidden">
                  <div className="absolute top-0 left-0 w-full h-1 bg-accent/20 blur-sm animate-scan" />
                  <div className="absolute inset-0 bg-gradient-to-b from-transparent via-accent/5 to-transparent opacity-30 animate-pulse" />
                </div>

                <div className="p-16 rounded-[4rem] bg-white/[0.02] border border-white/5 backdrop-blur-3xl ring-1 ring-white/5 relative group animate-float">
                  <div className="absolute inset-0 bg-primary/10 blur-[80px] rounded-full scale-90 group-hover:scale-110 transition-transform duration-1000 animate-pulse" />
                  <div className="absolute -inset-4 border border-white/5 rounded-[4.5rem] animate-[ping_3s_linear_infinite] opacity-20" />
                  <Video className="w-20 h-20 text-white/10 relative z-10 group-hover:text-accent/40 transition-colors duration-500" />
                </div>

                <div className="space-y-4 relative z-10">
                  <div className="flex items-center justify-center gap-3">
                    <div className="w-2 h-2 rounded-full bg-accent animate-pulse shadow-[0_0_10px_rgba(20,255,210,0.8)]" />
                    <p className="text-2xl font-black tracking-widest text-white/80 uppercase italic">Awaiting Interface</p>
                  </div>
                  <p className="text-muted-foreground/40 text-[10px] font-mono tracking-[0.4em] animate-pulse">CONNECT HARDWARE TO INITIATE SCAN</p>
                </div>
              </div>
            )}

            {isLoading && (
              <div className="absolute inset-0 bg-background/90 z-20 flex flex-col items-center justify-center backdrop-blur-2xl">
                <div className="relative">
                  <RefreshCw className="w-20 h-20 text-primary animate-spin mb-8" />
                  <div className="absolute inset-0 bg-primary/30 blur-[60px]" />
                </div>
                <p className="font-black text-xs tracking-[0.4em] text-primary uppercase animate-pulse">Establishing Neural Link</p>
              </div>
            )}
            
            {isStreaming && (
              <div className="absolute bottom-4 left-4 z-30 flex gap-2">
                <Badge variant="outline" className="bg-black/40 backdrop-blur-md border-white/10 text-[10px] font-mono py-1.5 px-3 rounded-full flex gap-2 items-center">
                  <Gauge className="w-3 h-3 text-accent" />
                  <span className="text-accent">{fps} FPS</span>
                </Badge>
                <Badge variant="outline" className="bg-black/40 backdrop-blur-md border-white/10 text-[10px] font-mono py-1.5 px-3 rounded-full flex gap-2 items-center">
                  <Cpu className="w-3 h-3 text-primary" />
                  <span className="text-primary">{inferenceTime}ms INF</span>
                </Badge>
              </div>
            )}
          </div>
        </CardContent>

        <CardFooter className="flex flex-col gap-12 p-12 bg-white/[0.01]">
          <div className="flex flex-col xl:flex-row items-end justify-between w-full gap-10">
            <div className="flex flex-col gap-6 w-full xl:w-auto">
              <Label className="text-[10px] font-black uppercase tracking-[0.3em] text-muted-foreground/40 ml-1">Stream Input Matrix</Label>
              <Tabs value={sourceMode} onValueChange={(val) => { stopStream(); setSourceMode(val as any); }} className="w-full sm:w-[360px]">
                <TabsList className="grid w-full grid-cols-2 h-16 bg-white/[0.03] border border-white/5 p-1.5 rounded-2xl">
                  <TabsTrigger value="camera" className="rounded-xl gap-3 text-xs font-black uppercase tracking-widest data-[state=active]:bg-primary data-[state=active]:text-white transition-all">
                    <Camera className="w-4 h-4" /> Live
                  </TabsTrigger>
                  <TabsTrigger value="file" className="rounded-xl gap-3 text-xs font-black uppercase tracking-widest data-[state=active]:bg-primary data-[state=active]:text-white transition-all">
                    <Upload className="w-4 h-4" /> Asset
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </div>

            <div className="flex flex-wrap items-center gap-6 justify-center xl:justify-end w-full xl:w-auto">
              <div className="flex items-center space-x-10 bg-white/[0.03] px-10 py-5 rounded-[2rem] border border-white/5">
                <div className="flex items-center gap-5">
                  <div className={cn("p-2.5 rounded-xl transition-all duration-500", isAutoZoomEnabled ? "bg-accent/15 text-accent shadow-[0_0_15px_rgba(20,255,210,0.2)]" : "bg-white/5 text-muted-foreground/40")}>
                    <Maximize className="w-5 h-5" />
                  </div>
                  <div className="flex flex-col">
                    <Label htmlFor="auto-zoom" className="text-[10px] font-black uppercase tracking-[0.2em] cursor-pointer mb-1">Cinematic Zoom</Label>
                    <Switch id="auto-zoom" checked={isAutoZoomEnabled} onCheckedChange={setIsAutoZoomEnabled} disabled={!isStreaming} className="data-[state=checked]:bg-accent" />
                  </div>
                </div>
                
                <div className="w-px h-10 bg-white/5" />

                <div className="flex items-center gap-5">
                  <div className={cn("p-2.5 rounded-xl transition-all duration-500", isLowLight ? "bg-primary/15 text-primary shadow-[0_0_15px_rgba(110,140,255,0.2)]" : "bg-white/5 text-muted-foreground/40")}>
                    {isLowLight ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
                  </div>
                  <div className="flex flex-col">
                    <Label htmlFor="low-light" className="text-[10px] font-black uppercase tracking-[0.2em] cursor-pointer mb-1">Enhancement</Label>
                    <Switch id="low-light" checked={isLowLight} onCheckedChange={setIsLowLight} disabled={!isStreaming} className="data-[state=checked]:bg-primary" />
                  </div>
                </div>
              </div>

              {!isStreaming ? (
                sourceMode === 'camera' ? (
                  <Button onClick={startCamera} disabled={isLoading || isModelLoading} className="h-16 px-12 rounded-[1.5rem] text-xs font-black uppercase tracking-[0.3em] transition-all hover:scale-[1.03] active:scale-95 bg-primary hover:bg-primary/80 shadow-2xl shadow-primary/20">
                    <Camera className="mr-4 h-5 w-5" /> Initialize Feed
                  </Button>
                ) : (
                  <Button onClick={() => fileInputRef.current?.click()} disabled={isLoading || isModelLoading} className="h-16 px-12 rounded-[1.5rem] text-xs font-black uppercase tracking-[0.3em] transition-all hover:scale-[1.03] active:scale-95 bg-primary hover:bg-primary/80 shadow-2xl shadow-primary/20">
                    <Upload className="mr-4 h-5 w-5" /> Select Resource
                  </Button>
                )
              ) : (
                <Button variant="destructive" onClick={stopStream} className="h-16 px-12 rounded-[1.5rem] text-xs font-black uppercase tracking-[0.3em] transition-all hover:scale-[1.03] active:scale-95 bg-red-500/10 hover:bg-red-500/20 text-red-500 border border-red-500/20">
                  <CameraOff className="mr-4 h-5 w-5" /> Kill Stream
                </Button>
              )}
            </div>
          </div>

          <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="video/mp4" className="hidden" />

          {hasCameraPermission === false && (
            <Alert className="border-red-500/20 bg-red-500/5 text-red-500 rounded-[2rem] p-6">
              <AlertCircle className="h-5 w-5" />
              <AlertTitle className="font-black text-[11px] uppercase tracking-[0.2em] mb-2">Hardware Access Denied</AlertTitle>
              <AlertDescription className="text-sm opacity-70 font-medium">
                Camera initialization failed. System security policies may be blocking hardware access.
              </AlertDescription>
            </Alert>
          )}
        </CardFooter>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        {[
          { title: "Neural Lock", desc: "Proprietary IoU tracking maintains subject persistence across complex occlusion.", icon: <Activity className="w-6 h-6" /> },
          { title: "Smart Optic", desc: "Interpolated viewport scaling ensures smooth, jitter-free subject centering.", icon: <Maximize className="w-6 h-6" /> },
          { title: "Digital Bokeh", desc: "Real-time Gaussian isolation provides selective focus on prioritized targets.", icon: <Sparkles className="w-6 h-6" /> }
        ].map((feature, i) => (
          <div key={i} className="glass-panel p-10 rounded-[2.5rem] space-y-6 hover:bg-white/[0.04] transition-all duration-500 group relative overflow-hidden">
            <div className="absolute top-0 right-0 p-4 opacity-0 group-hover:opacity-100 transition-opacity">
              <Cpu className="w-10 h-10 text-primary/10" />
            </div>
            <div className="p-4 bg-primary/10 text-primary w-fit rounded-[1.25rem] border border-primary/10 group-hover:scale-110 group-hover:bg-primary/20 transition-all duration-500">
              {feature.icon}
            </div>
            <div className="space-y-3">
              <h3 className="text-base font-black uppercase tracking-[0.2em] text-white/90">{feature.title}</h3>
              <p className="text-sm text-muted-foreground/50 leading-relaxed font-medium">
                {feature.desc}
              </p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};