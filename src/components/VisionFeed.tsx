"use client"

import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Camera, CameraOff, Sparkles, RefreshCw, AlertCircle, Box, Cpu, MousePointer2, XCircle, Moon, Sun, Upload, Video, Maximize, Search } from 'lucide-react';
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

  // Fix hydration mismatch
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
        console.error("Error loading model:", err);
        toast({
          title: "AI Model Failed",
          description: "Could not initialize object detection.",
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
        title: "Camera Access Failed",
        description: "Please check your browser's camera permissions.",
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
        title: "Video Loading Failed",
        description: "Could not load the selected video file.",
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
      selectedHistoryRef.current = []; // Reset history for new focus
      trackingConfidenceRef.current = 1.0; // Perfect match on click
      reacquisitionCountRef.current = 0;
      setIsReacquiring(false);
      setSelectedLabel(clickedObj.class);
      toast({ title: `Focus Locked: ${clickedObj.class}` });
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

      // Detection Loop (Throttled)
      // If reacquiring, run detection more aggressively (every 3 frames instead of 10)
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
            
            // Smart reacquisition logic
            if (maxIoU > 0.3 && bestMatch) { 
              selectedObjectRef.current = bestMatch;
              trackingConfidenceRef.current = maxIoU;
              reacquisitionCountRef.current = 0;
              setIsReacquiring(false);
              
              // Add to motion trail history
              selectedHistoryRef.current.push([...bestMatch.bbox]);
              if (selectedHistoryRef.current.length > 5) {
                selectedHistoryRef.current.shift();
              }
            } else {
              reacquisitionCountRef.current++;
              // If IoU is low for 3 frames, flag reacquisition
              if (reacquisitionCountRef.current >= 3) {
                setIsReacquiring(true);
                trackingConfidenceRef.current = 0;
                
                // If extremely lost (e.g., class not found at all), or after extended failure, we might reset
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
      
      // Cinematic Background Blur
      ctx.filter = `${enhancementFilter}blur(12px) brightness(0.7)`;
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

      // Reset filter for sharp overlays
      ctx.filter = "none";

      // Draw Motion Trails for selected subject
      if (activeSelection && !isReacquiring && selectedHistoryRef.current.length > 1) {
        selectedHistoryRef.current.forEach((trailBbox, index) => {
          if (index === selectedHistoryRef.current.length - 1) return;
          const alpha = (index + 1) / selectedHistoryRef.current.length * 0.4;
          ctx.strokeStyle = `rgba(34, 197, 94, ${alpha})`;
          ctx.lineWidth = 2;
          ctx.strokeRect(trailBbox[0], trailBbox[1], trailBbox[2], trailBbox[3]);
        });
      }

      // Render Detection Boxes
      predictionsRef.current.forEach(prediction => {
        const [x, y, width, height] = prediction.bbox;
        const isSelected = activeSelection && prediction.class === activeSelection.class && calculateIoU(prediction.bbox, activeSelection.bbox) > 0.8;
        
        const primaryColor = isSelected ? '#22c55e' : 'rgba(239, 68, 68, 0.4)';
        const labelBg = isSelected ? 'rgba(21, 128, 61, 0.9)' : 'rgba(185, 28, 28, 0.5)';

        ctx.strokeStyle = primaryColor;
        ctx.lineWidth = isSelected ? 4 : 2;
        ctx.strokeRect(x, y, width, height);

        let labelText = `${prediction.class} ${Math.round(prediction.score * 100)}%`;
        if (isSelected) {
          const conf = Math.round(trackingConfidenceRef.current * 100);
          labelText = `${prediction.class} (Tracking: ${conf}%)`;
        }
        
        ctx.font = `bold ${Math.max(14, canvas.width * 0.012)}px sans-serif`;
        const textWidth = ctx.measureText(labelText).width;
        ctx.fillStyle = labelBg;
        ctx.fillRect(x, y - (canvas.width * 0.03), textWidth + 12, canvas.width * 0.03);
        ctx.fillStyle = 'white';
        ctx.fillText(labelText, x + 6, y - (canvas.width * 0.008));
      });

      // Show "Reacquiring target..." indicator on canvas
      if (isReacquiring && activeSelection) {
        const [x, y, w, h] = activeSelection.bbox;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(x, y + h / 2 - 20, w, 40);
        ctx.fillStyle = '#facc15';
        ctx.font = 'bold 16px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Reacquiring target...', x + w / 2, y + h / 2 + 5);
        ctx.textAlign = 'start'; // reset
      }

      ctx.restore();
      animationFrameId = requestAnimationFrame(render);
    };

    if (isStreaming) animationFrameId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animationFrameId);
  }, [isStreaming, model, isLowLight, isAutoZoomEnabled, isReacquiring]);

  if (!isMounted) return null;

  return (
    <div className="w-full max-w-5xl mx-auto p-4 md:p-10 space-y-10 animate-fade-in">
      <Card className="overflow-hidden shadow-[0_20px_50px_rgba(0,0,0,0.1)] border-none bg-white/90 backdrop-blur-md rounded-2xl">
        <CardHeader className="text-center pb-6 border-b border-muted/50 relative">
          {selectedLabel && (
            <div className="absolute top-6 right-6 animate-scale-in flex flex-col items-end gap-2">
              <Badge variant="default" className={cn(
                "gap-2 pl-3 py-1.5 pr-2 shadow-xl border-none transition-colors duration-500",
                isReacquiring ? "bg-yellow-600 animate-pulse" : "bg-green-600 hover:bg-green-700"
              )}>
                {isReacquiring ? <Search className="w-4 h-4 animate-spin" /> : null}
                {isReacquiring ? `Reacquiring: ${selectedLabel}` : `Locked: ${selectedLabel}`}
                <Button variant="ghost" size="icon" className="h-5 w-5 rounded-full hover:bg-white/20 p-0" onClick={() => { selectedObjectRef.current = null; selectedHistoryRef.current = []; trackingConfidenceRef.current = 0; reacquisitionCountRef.current = 0; setIsReacquiring(false); setSelectedLabel(null); }}>
                  <XCircle className="w-4 h-4" />
                </Button>
              </Badge>
            </div>
          )}
          
          <div className="flex justify-center mb-4">
            {isModelLoading ? (
              <Badge variant="secondary" className="animate-pulse gap-2 px-4 py-1">
                <RefreshCw className="w-3.5 h-3.5 animate-spin" /> Preparing AI Engine...
              </Badge>
            ) : (
              <Badge variant="outline" className="text-primary border-primary/30 gap-2 px-4 py-1 bg-primary/5">
                <Cpu className="w-3.5 h-3.5" /> Neural Network Active
              </Badge>
            )}
          </div>
          
          <CardTitle className="text-4xl md:text-5xl font-bold tracking-tight text-primary flex flex-col md:flex-row items-center justify-center gap-4">
            <div className="p-3 bg-primary/10 rounded-2xl">
              <Sparkles className="w-10 h-10 text-primary" />
            </div>
            Vision Canvas: Cinematic AI Focus
          </CardTitle>
          <CardDescription className="text-lg max-w-2xl mx-auto mt-4 leading-relaxed">
            Real-time object segmentation with **Cinematic Auto-Zoom**, **Motion Trails**, and **Smart Reacquisition**. Click any subject to lock focus and track smoothly.
            <div className="mt-2 text-sm font-medium text-muted-foreground bg-muted/30 py-2 rounded-lg px-4 border border-muted/50">
              Interpolated tracking ensures jitter-free camera motion and selective depth-of-field.
            </div>
          </CardDescription>
        </CardHeader>
        
        <CardContent className="p-0 relative group">
          <div className={cn(
            "relative aspect-video bg-slate-900 flex flex-col items-center justify-center overflow-hidden transition-all duration-700",
            !isStreaming && "bg-slate-50"
          )}>
            <video ref={videoRef} autoPlay playsInline muted className="absolute inset-0 w-full h-full object-cover opacity-0 pointer-events-none" />
            <canvas ref={canvasRef} onClick={handleCanvasClick} className={cn(
              "absolute inset-0 z-10 transition-opacity duration-1000 w-full h-full object-cover cursor-crosshair",
              isStreaming ? "opacity-100" : "opacity-0 pointer-events-none"
            )} />

            {!isStreaming && !isLoading && (
              <div className="z-10 flex flex-col items-center gap-6 animate-scale-in text-center px-6">
                <div className="p-10 rounded-[2.5rem] bg-white shadow-2xl border border-muted ring-4 ring-primary/5">
                  <Video className="w-20 h-20 text-primary/40" />
                </div>
                <div className="space-y-2">
                  <p className="text-2xl font-bold text-slate-800">Ready to Visualize</p>
                  <p className="text-muted-foreground">Choose a source below to start the intelligent stream</p>
                </div>
              </div>
            )}

            {isLoading && (
              <div className="absolute inset-0 bg-white/60 z-20 flex flex-col items-center justify-center backdrop-blur-md">
                <RefreshCw className="w-16 h-16 text-primary animate-spin mb-4" />
                <p className="font-bold text-primary animate-pulse">Initializing Stream...</p>
              </div>
            )}
          </div>
        </CardContent>

        <CardFooter className="flex flex-col gap-8 p-10 bg-gradient-to-b from-transparent to-muted/20 border-t border-muted/50">
          <div className="flex flex-col lg:flex-row items-center justify-between w-full gap-8">
            <div className="flex flex-col gap-4 w-full lg:w-auto">
              <Label className="text-xs font-bold uppercase tracking-widest text-muted-foreground ml-1">Stream Source</Label>
              <Tabs value={sourceMode} onValueChange={(val) => { stopStream(); setSourceMode(val as any); }} className="w-full sm:w-[300px]">
                <TabsList className="grid w-full grid-cols-2 h-12 bg-white border border-muted shadow-sm">
                  <TabsTrigger value="camera" className="gap-2 data-[state=active]:bg-primary data-[state=active]:text-white">
                    <Camera className="w-4 h-4" /> Camera
                  </TabsTrigger>
                  <TabsTrigger value="file" className="gap-2 data-[state=active]:bg-primary data-[state=active]:text-white">
                    <Upload className="w-4 h-4" /> MP4 File
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </div>

            <div className="flex flex-col gap-4 w-full lg:w-auto items-center lg:items-end">
              <Label className="text-xs font-bold uppercase tracking-widest text-muted-foreground mr-1">Cinematic Controls</Label>
              <div className="flex flex-wrap items-center gap-4">
                <div className="flex items-center space-x-6 bg-white px-5 py-3 rounded-2xl border border-muted shadow-sm">
                  <div className="flex items-center gap-3">
                    <Maximize className={cn("w-5 h-5", isAutoZoomEnabled ? "text-primary" : "text-muted-foreground")} />
                    <Label htmlFor="auto-zoom" className="text-sm font-bold cursor-pointer">Auto-Zoom</Label>
                  </div>
                  <Switch id="auto-zoom" checked={isAutoZoomEnabled} onCheckedChange={setIsAutoZoomEnabled} disabled={!isStreaming} />
                  
                  <div className="w-px h-6 bg-muted mx-2" />

                  <div className="flex items-center gap-3">
                    {isLowLight ? <Moon className="w-5 h-5 text-indigo-600" /> : <Sun className="w-5 h-5 text-amber-500" />}
                    <Label htmlFor="low-light" className="text-sm font-bold cursor-pointer">Low Light</Label>
                  </div>
                  <Switch id="low-light" checked={isLowLight} onCheckedChange={setIsLowLight} disabled={!isStreaming} />
                </div>

                {!isStreaming ? (
                  sourceMode === 'camera' ? (
                    <Button size="lg" onClick={startCamera} disabled={isLoading || isModelLoading} className="px-12 h-16 rounded-2xl text-xl font-bold transition-all hover:scale-105 shadow-2xl shadow-primary/30">
                      <Camera className="mr-3 h-6 w-6" /> Start Stream
                    </Button>
                  ) : (
                    <Button size="lg" onClick={() => fileInputRef.current?.click()} disabled={isLoading || isModelLoading} className="px-12 h-16 rounded-2xl text-xl font-bold transition-all hover:scale-105 shadow-2xl shadow-primary/30">
                      <Upload className="mr-3 h-6 w-6" /> Select MP4
                    </Button>
                  )
                ) : (
                  <Button variant="destructive" size="lg" onClick={stopStream} className="px-12 h-16 rounded-2xl text-xl font-bold transition-all hover:scale-105 shadow-2xl shadow-destructive/30">
                    <CameraOff className="mr-3 h-6 w-6" /> Stop Stream
                  </Button>
                )}
              </div>
            </div>
          </div>

          <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="video/mp4" className="hidden" />

          {hasCameraPermission === false && (
            <Alert variant="destructive" className="border-none bg-red-50 text-red-900 rounded-xl shadow-lg">
              <AlertCircle className="h-5 w-5" />
              <AlertTitle className="font-bold">Hardware Access Required</AlertTitle>
              <AlertDescription>
                Camera access was blocked. Please enable it in your browser address bar to use this feature.
              </AlertDescription>
            </Alert>
          )}
        </CardFooter>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-8 animate-fade-in delay-300">
        {[
          { title: "Smart Reacquisition", desc: "If tracking is lost, the AI enters an aggressive search mode to re-lock the subject.", icon: <Search className="w-8 h-8 text-primary" /> },
          { title: "Cinematic Auto-Zoom", desc: "AI calculates subject prominence and applies a 1.5x interpolated scale transition.", icon: <Maximize className="w-8 h-8 text-primary" /> },
          { title: "Selective Clarity", desc: "Gaussian blur masks isolate your subject from the background for depth of field.", icon: <MousePointer2 className="w-8 h-8 text-primary" /> }
        ].map((feature, i) => (
          <Card key={i} className="bg-white/70 border-none shadow-xl hover:shadow-2xl transition-all duration-300 rounded-2xl p-2 hover:-translate-y-1">
            <CardHeader className="pb-4">
              <div className="mb-4 p-3 bg-primary/5 w-fit rounded-xl border border-primary/10">{feature.icon}</div>
              <CardTitle className="text-xl font-bold">{feature.title}</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-muted-foreground leading-relaxed font-medium">
                {feature.desc}
              </p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  );
};