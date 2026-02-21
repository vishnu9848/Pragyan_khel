"use client"

import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Camera, CameraOff, RefreshCw, AlertCircle, Cpu, XCircle, Upload, Video, Maximize, Activity, Gauge, Sparkles, Target, Zap, Globe } from 'lucide-react';
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

/** Minimal type for BodyPix PersonSegmentation (avoids loading body-pix at module parse time). */
interface PersonSegmentationLike {
  data: Uint8Array;
  width: number;
  height: number;
  pose: unknown;
}

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

/** Mask overlap: sum of min(a1,a2) / sum(a1). Returns 0 if prev has no pixels. */
function maskOverlap(prev: ImageData, curr: ImageData): number {
  if (prev.data.length !== curr.data.length) return 0;
  let sumPrev = 0, sumMin = 0;
  for (let i = 3; i < prev.data.length; i += 4) {
    const a1 = prev.data[i];
    const a2 = curr.data[i];
    sumPrev += a1;
    sumMin += Math.min(a1, a2);
  }
  return sumPrev > 0 ? sumMin / sumPrev : 0;
}

/** Check if mask ImageData has alpha > threshold at (px, py). */
function maskContainsPoint(maskData: ImageData, px: number, py: number, width: number, height: number, threshold = 128): boolean {
  const x = Math.floor(px);
  const y = Math.floor(py);
  if (x < 0 || x >= width || y < 0 || y >= height) return false;
  const i = (y * width + x) * 4 + 3;
  return maskData.data[i] > threshold;
}

/** Convert BodyPix PersonSegmentation (1/0 data) to ImageData (foreground alpha 255). */
function personSegToImageData(seg: PersonSegmentationLike): ImageData {
  const { data, width, height } = seg;
  const out = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < data.length; i++) {
    out[i * 4 + 3] = data[i] ? 255 : 0;
  }
  return new ImageData(out, width, height);
}

export const VisionFeed: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  
  const predictionsRef = useRef<cocoSsd.DetectedObject[]>([]);
  const frameCountRef = useRef(0);
  const isDetectingRef = useRef(false);
  const selectedObjectRef = useRef<cocoSsd.DetectedObject | null>(null);

  type BodyPixNet = { segmentMultiPerson(input: unknown): Promise<PersonSegmentationLike[]> };
  const segmentationsRef = useRef<PersonSegmentationLike[]>([]);
  const selectedInstanceIdRef = useRef<number | null>(null);
  const previousMaskDataRef = useRef<ImageData | null>(null);
  const selectedMaskImageDataRef = useRef<ImageData | null>(null);
  const isSegmentingRef = useRef(false);
  
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
  const [segmenter, setSegmenter] = useState<BodyPixNet | null>(null);
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
    const loadModels = async () => {
      try {
        setIsModelLoading(true);
        await tf.ready();
        const loadedModel = await cocoSsd.load({ base: 'mobilenet_v2' });
        setModel(loadedModel);
        try {
          const bodyPix = await import('@tensorflow-models/body-pix');
          const loadedSegmenter = await bodyPix.load({
            architecture: 'MobileNetV1',
            outputStride: 16,
            multiplier: 0.75,
            quantBytes: 2,
          });
          setSegmenter(loadedSegmenter as BodyPixNet);
        } catch (_) {
          setSegmenter(null);
        }
      } catch (err) {
        toast({ title: "Inference Error", description: "Could not load AI vision core.", variant: "destructive" });
      } finally { setIsModelLoading(false); }
    };
    loadModels();
  }, [toast]);

  const stopStream = useCallback(() => {
    if (stream) stream.getTracks().forEach(track => track.stop());
    if (videoFileUrl) URL.revokeObjectURL(videoFileUrl);
    setStream(null);
    setVideoFileUrl(null);
    setIsStreaming(false);
    predictionsRef.current = [];
    selectedObjectRef.current = null;
    selectedInstanceIdRef.current = null;
    previousMaskDataRef.current = null;
    selectedMaskImageDataRef.current = null;
    segmentationsRef.current = [];
    setSelectedLabel(null);
    zoomFactorRef.current = 1;
    focusAlphaRef.current = 0;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
      videoRef.current.src = '';
    }
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
    const video = videoRef.current;
    if (!video) {
      setIsLoading(false);
      return;
    }
    try {
      // Clear any previous camera stream so file source takes over
      video.srcObject = null;
      video.removeAttribute('src');
      const url = URL.createObjectURL(file);
      setVideoFileUrl(url);
      setSourceMode('file');

      await new Promise<void>((resolve, reject) => {
        const onReady = () => {
          video.removeEventListener('loadeddata', onReady);
          video.removeEventListener('error', onError);
          resolve();
        };
        const onError = () => {
          video.removeEventListener('loadeddata', onReady);
          video.removeEventListener('error', onError);
          reject(new Error('Video failed to load'));
        };
        video.addEventListener('loadeddata', onReady, { once: true });
        video.addEventListener('error', onError, { once: true });
        video.src = url;
        video.loop = true;
        video.muted = true;
        video.load();
      });

      await video.play();
      setIsStreaming(true);
    } catch (err) {
      toast({ title: "Load Error", description: "Failed to load video file. Try MP4 or WebM.", variant: "destructive" });
    } finally {
      setIsLoading(false);
    }
    e.target.value = '';
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

    const segs = segmentationsRef.current;
    if (segmenter && segs.length > 0) {
      let foundId: number | null = null;
      for (let i = 0; i < segs.length; i++) {
        const maskData = personSegToImageData(segs[i]);
        if (maskContainsPoint(maskData, x, y, canvas.width, canvas.height, 64)) {
          foundId = i;
          break;
        }
      }
      selectedInstanceIdRef.current = foundId;
      if (foundId !== null) {
        previousMaskDataRef.current = personSegToImageData(segs[foundId]);
        selectedMaskImageDataRef.current = previousMaskDataRef.current;
        setSelectedLabel('Person');
        focusAlphaRef.current = 1;
        selectedObjectRef.current = null;
        focusBboxRef.current = null;
      } else {
        setSelectedLabel(null);
      }
      return;
    }

    const clicked = predictionsRef.current.find(p => {
      const [bx, by, bw, bh] = p.bbox;
      return x >= bx && x <= bx + bw && y >= by && y <= by + bh;
    });

    if (clicked) {
      selectedObjectRef.current = JSON.parse(JSON.stringify(clicked));
      setSelectedLabel(clicked.class);
      focusBboxRef.current = [...clicked.bbox];
      focusAlphaRef.current = 1;
      focusScaleRef.current = 0.6;
      selectedInstanceIdRef.current = null;
      previousMaskDataRef.current = null;
    } else {
      selectedObjectRef.current = null;
      setSelectedLabel(null);
      selectedInstanceIdRef.current = null;
      previousMaskDataRef.current = null;
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
        model.detect(video, 30, 0.55).then(predictions => {
          detectionTimeRef.current = Math.round(performance.now() - start);
          predictionsRef.current = predictions;
          const current = selectedObjectRef.current;
          if (current) {
            const best = predictions.filter(p => p.class === current.class)
              .sort((a, b) => calculateIoU(b.bbox, current.bbox) - calculateIoU(a.bbox, current.bbox))[0];
            
            if (best && calculateIoU(best.bbox, current.bbox) > 0.5) {
              selectedObjectRef.current = best;
              setIsReacquiring(false);
            } else {
              setIsReacquiring(true);
            }
          }
          isDetectingRef.current = false;
        }).catch(() => isDetectingRef.current = false);
      }

      if (segmenter && !isSegmentingRef.current && frameCountRef.current % 6 === 0) {
        isSegmentingRef.current = true;
        segmenter.segmentMultiPerson(video).then((segs) => {
          segmentationsRef.current = segs;
          const sid = selectedInstanceIdRef.current;
          const prev = previousMaskDataRef.current;
          if (sid !== null && segs.length > 0) {
            let targetIdx = sid < segs.length ? sid : 0;
            if (prev && segs.length > 1) {
              const maskDataArr = segs.map((s) => personSegToImageData(s));
              let bestOverlap = 0;
              for (let i = 0; i < maskDataArr.length; i++) {
                const o = maskOverlap(prev, maskDataArr[i]);
                if (o > bestOverlap) {
                  bestOverlap = o;
                  targetIdx = i;
                }
              }
              if (bestOverlap < 0.25) {
                selectedInstanceIdRef.current = null;
                previousMaskDataRef.current = null;
                selectedMaskImageDataRef.current = null;
                setSelectedLabel(null);
              } else {
                selectedInstanceIdRef.current = targetIdx;
                previousMaskDataRef.current = maskDataArr[targetIdx];
                selectedMaskImageDataRef.current = maskDataArr[targetIdx];
              }
            } else {
              selectedInstanceIdRef.current = targetIdx;
              previousMaskDataRef.current = personSegToImageData(segs[targetIdx]);
              selectedMaskImageDataRef.current = previousMaskDataRef.current;
            }
          } else {
            selectedMaskImageDataRef.current = null;
          }
          isSegmentingRef.current = false;
        }).catch(() => { isSegmentingRef.current = false; });
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
        focusAlphaRef.current = lerp(focusAlphaRef.current, 1.0, 0.2);
        focusScaleRef.current = lerp(focusScaleRef.current, 0.96, 0.1);
        if (isAutoZoomEnabled) {
          targetZoom = 1.8;
          targetPanX = focusBboxRef.current[0] + focusBboxRef.current[2] / 2;
          targetPanY = focusBboxRef.current[1] + focusBboxRef.current[3] / 2;
        }
      } else {
        if (!active) {
          focusAlphaRef.current = lerp(focusAlphaRef.current, 0, 0.1);
          focusScaleRef.current = lerp(focusScaleRef.current, 0.8, 0.1);
        }
      }

      zoomFactorRef.current = lerp(zoomFactorRef.current, targetZoom, 0.08);
      panXRef.current = lerp(panXRef.current, targetPanX, 0.08);
      panYRef.current = lerp(panYRef.current, targetPanY, 0.08);

      ctx.save();
      ctx.translate(canvas.width / 2, canvas.height / 2);
      ctx.scale(zoomFactorRef.current, zoomFactorRef.current);
      ctx.translate(-panXRef.current, -panYRef.current);

      const maskData = selectedMaskImageDataRef.current;
      const useInstanceFocus = selectedInstanceIdRef.current !== null && maskData;

      if (useInstanceFocus) {
        const blurPx = 36;
        const grayAmount = 0.55;
        ctx.save();
        ctx.filter = `blur(${blurPx}px) grayscale(${grayAmount})`;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.restore();
        ctx.save();
        ctx.filter = "none";
        const off = document.createElement('canvas');
        off.width = canvas.width;
        off.height = canvas.height;
        const offCtx = off.getContext('2d')!;
        offCtx.drawImage(video, 0, 0);
        offCtx.globalCompositeOperation = 'destination-in';
        const maskCanvas = document.createElement('canvas');
        maskCanvas.width = maskData.width;
        maskCanvas.height = maskData.height;
        const maskCtx = maskCanvas.getContext('2d')!;
        maskCtx.putImageData(maskData, 0, 0);
        offCtx.drawImage(maskCanvas, 0, 0);
        ctx.drawImage(off, 0, 0);
        ctx.restore();
      } else if (focusAlphaRef.current > 0.01 && focusBboxRef.current) {
        const [fx, fy, fw, fh] = focusBboxRef.current;
        const s = Math.min(focusScaleRef.current, 0.96);
        const maxW = 0.5 * canvas.width;
        const maxH = 0.5 * canvas.height;
        const dw = Math.min(fw * s, maxW);
        const dh = Math.min(fh * s, maxH);
        const cx = fx + fw / 2;
        const cy = fy + fh / 2;
        let dx = cx - dw / 2;
        let dy = cy - dh / 2;
        dx = Math.max(0, Math.min(dx, canvas.width - dw));
        dy = Math.max(0, Math.min(dy, canvas.height - dh));

        ctx.save();
        const blurPx = focusAlphaRef.current * 36;
        const grayAmount = focusAlphaRef.current * 0.55;
        ctx.filter = `blur(${blurPx}px) grayscale(${grayAmount})`;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.restore();

        ctx.save();
        ctx.filter = "none";
        ctx.drawImage(video, dx, dy, dw, dh, dx, dy, dw, dh);
        ctx.restore();

        ctx.strokeStyle = '#10b981';
        ctx.lineWidth = 5;
        ctx.setLineDash([]);
        ctx.strokeRect(dx, dy, dw, dh);

        if (active) {
          const label = `${active.class.toUpperCase()} ${Math.round(active.score * 100)}%`;
          ctx.font = 'bold 14px Inter, system-ui, sans-serif';
          const textWidth = ctx.measureText(label).width;
          ctx.fillStyle = '#10b981';
          ctx.fillRect(dx - 2, dy - 32, textWidth + 20, 32);
          ctx.fillStyle = '#ffffff';
          ctx.fillText(label, dx + 8, dy - 10);
        }

        ctx.shadowColor = 'rgba(16, 185, 129, 0.5)';
        ctx.shadowBlur = 14;
        ctx.strokeRect(dx, dy, dw, dh);
        ctx.shadowBlur = 0;
      } else {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      }

      if (!useInstanceFocus && focusAlphaRef.current < 0.5) {
        predictionsRef.current.forEach(p => {
          if (active && p.class === active.class) return;
          const [px, py, pw, ph] = p.bbox;
          ctx.strokeStyle = 'rgba(16, 185, 129, 0.2)';
          ctx.setLineDash([5, 5]);
          ctx.strokeRect(px, py, pw, ph);
          ctx.font = 'bold 10px Inter, system-ui, sans-serif';
          const label = `${p.class.toUpperCase()} ${Math.round(p.score * 100)}%`;
          ctx.fillStyle = 'rgba(16, 185, 129, 0.6)';
          ctx.fillText(label, px + 4, py + 14);
        });
      }

      ctx.restore();
      animationFrameId = requestAnimationFrame(render);
    };
    if (isStreaming) animationFrameId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animationFrameId);
  }, [isStreaming, model, segmenter, isLowLight, isAutoZoomEnabled, isReacquiring]);

  if (!isMounted) return null;

  return (
    <div className="w-full space-y-8 animate-fade-in">
      <Card className="glass-dark overflow-hidden rounded-2xl border border-white/10 shadow-2xl">
        <CardHeader className="text-center py-10 border-b border-white/10 relative">
          {selectedLabel && (
            <div className="absolute top-8 right-8 animate-scale-in z-30">
              <Badge className={cn(
                "gap-3 pl-4 py-2 pr-2 border-none transition-all rounded-full font-bold",
                isReacquiring ? "bg-amber-500/20 text-amber-400 border border-amber-500/30 animate-pulse" : "bg-primary/20 text-primary border border-primary/30 emerald-glow"
              )}>
                <span className="text-[10px] uppercase tracking-widest">{isReacquiring ? "Seeking" : "Locked"}</span>
                <span className="text-sm px-1">
                  {selectedLabel} {selectedObjectRef.current ? `(${Math.round(selectedObjectRef.current.score * 100)}%)` : ''}
                </span>
                <Button variant="ghost" size="icon" className="h-6 w-6 rounded-full hover:bg-white/10 text-foreground" onClick={() => { selectedObjectRef.current = null; selectedInstanceIdRef.current = null; previousMaskDataRef.current = null; selectedMaskImageDataRef.current = null; setSelectedLabel(null); }}>
                  <XCircle className="w-4 h-4 opacity-60" />
                </Button>
              </Badge>
            </div>
          )}
          
          <div className="flex justify-center mb-6">
            <Badge variant="outline" className={cn(
              "gap-2.5 px-6 py-1.5 rounded-full text-[10px] uppercase tracking-widest font-bold border-white/20",
              isModelLoading ? "text-muted-foreground" : "text-primary border-primary/30 bg-primary/10"
            )}>
              {isModelLoading ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Cpu className="w-3.5 h-3.5" />}
              {isModelLoading ? "Initializing Core" : "Neural Link Active"}
            </Badge>
          </div>
          
          <CardTitle className="font-heading text-4xl md:text-5xl font-extrabold tracking-tight text-foreground">
            Vision Canvas
          </CardTitle>
          <CardDescription className="text-muted-foreground max-w-lg mx-auto mt-4 text-sm font-medium leading-relaxed">
            <span className="text-primary font-bold">Instance-level focus</span>: click one person—only they stay sharp; everyone else and the background blur. Pixel mask, smooth boundary, tracking by ID.
          </CardDescription>
        </CardHeader>
        
        <CardContent className="p-0 relative bg-black/20">
          <div className={cn("relative aspect-video flex items-center justify-center overflow-hidden", !isStreaming && "opacity-100")}>
            <video ref={videoRef} autoPlay playsInline muted className="hidden" />
            <canvas ref={canvasRef} onClick={handleCanvasClick} className={cn(
              "absolute inset-0 z-10 w-full h-full object-cover cursor-crosshair transition-opacity duration-1000",
              isStreaming ? "opacity-100" : "opacity-0"
            )} />

            {!isStreaming && !isLoading && (
              <div className="z-10 flex flex-col items-center justify-center gap-12 animate-scale-in text-center p-12 w-full h-full min-h-[500px] relative">
                <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,hsl(var(--primary)/0.08),transparent_70%)] animate-pulse-subtle" />
                
                <div className="relative flex items-center justify-center w-80 h-80">
                  <div className="absolute inset-0 rounded-full border-[2px] border-dashed border-primary/30 animate-spin-slow" />
                  <div className="absolute inset-4 rounded-full border-[1px] border-primary/20 animate-spin-reverse-slow" />
                  <div className="absolute inset-8 rounded-full border-[4px] border-t-primary/40 border-r-transparent border-b-secondary/40 border-l-transparent animate-spin" />
                  
                  <div className="absolute -top-4 left-1/2 -translate-x-1/2 w-4 h-4 bg-primary rounded-full blur-[2px] animate-float" />
                  <div className="absolute bottom-10 right-10 w-3 h-3 bg-secondary rounded-full blur-[1px] animate-bounce" style={{ animationDuration: '3s' }} />
                  <div className="absolute top-20 left-0 w-2 h-2 bg-primary/60 rounded-full animate-pulse" />

                  <div className="relative p-12 rounded-full glass-panel border-primary/20 animate-pulse-subtle flex items-center justify-center group overflow-hidden emerald-glow">
                    <div className="absolute inset-0 bg-gradient-to-tr from-primary/10 to-secondary/10 opacity-0 group-hover:opacity-100 transition-opacity duration-700" />
                    <Globe className="w-24 h-24 text-primary relative z-10 animate-spin-slow" />
                    <Sparkles className="absolute top-8 right-8 w-6 h-6 text-secondary animate-pulse" />
                    <Target className="absolute bottom-8 left-8 w-6 h-6 text-primary/40" />
                  </div>
                </div>

                <div className="space-y-6 relative z-10">
                  <div className="flex flex-col items-center gap-2">
                    <p className="font-heading text-3xl font-bold tracking-[0.2em] text-foreground uppercase flex items-center gap-4">
                      <Zap className="w-8 h-8 text-primary animate-pulse" />
                      Interface Ready
                      <Activity className="w-8 h-8 text-secondary animate-pulse" />
                    </p>
                    <div className="h-1 w-48 bg-gradient-to-r from-transparent via-primary/40 to-transparent rounded-full" />
                  </div>
                  <div className="flex items-center justify-center gap-6">
                    <Badge variant="outline" className="border-primary/30 text-primary font-mono text-[10px] uppercase tracking-[0.2em] px-4 py-1 bg-primary/5">
                      Link Hardware
                    </Badge>
                    <div className="w-2 h-2 rounded-full bg-muted-foreground/40" />
                    <Badge variant="outline" className="border-secondary/30 text-secondary font-mono text-[10px] uppercase tracking-[0.2em] px-4 py-1 bg-secondary/5">
                      Neural Sync
                    </Badge>
                  </div>
                </div>
              </div>
            )}

            {isLoading && (
              <div className="absolute inset-0 bg-background/80 z-20 flex flex-col items-center justify-center backdrop-blur-sm">
                <RefreshCw className="w-12 h-12 text-primary animate-spin mb-4" />
                <p className="font-bold text-[10px] tracking-[0.4em] text-foreground uppercase">Linking Hardware</p>
              </div>
            )}
            
            {isStreaming && (
              <div className="absolute bottom-6 left-6 z-30 flex gap-2">
                <Badge className="glass-panel text-foreground text-[9px] font-mono py-1.5 px-3 rounded-full flex gap-2 items-center border-primary/20">
                  <Gauge className="w-3 h-3 text-primary" /> {fps} FPS
                </Badge>
                <Badge className="glass-panel text-foreground text-[9px] font-mono py-1.5 px-3 rounded-full flex gap-2 items-center border-primary/20">
                  <Cpu className="w-3 h-3 text-primary" /> {inferenceTime}ms INF
                </Badge>
              </div>
            )}
          </div>
        </CardContent>

        <CardFooter className="flex flex-col gap-10 p-10 border-t border-white/10 bg-card/50">
          <div className="flex flex-col lg:flex-row items-center justify-between w-full gap-8">
            <div className="flex flex-col gap-4 w-full lg:w-auto">
              <Label className="text-[9px] font-bold uppercase tracking-[0.3em] text-muted-foreground ml-1">Input Source</Label>
              <Tabs value={sourceMode} onValueChange={(v) => { stopStream(); setSourceMode(v as any); }} className="w-full sm:w-[320px]">
                <TabsList className="grid w-full grid-cols-2 h-12 bg-white/5 border border-white/10 p-1 rounded-xl">
                  <TabsTrigger value="camera" className="rounded-lg gap-2 text-[10px] font-bold uppercase tracking-widest data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:border-0 border border-transparent transition-all">
                    <Camera className="w-3.5 h-3.5" /> Camera
                  </TabsTrigger>
                  <TabsTrigger value="file" className="rounded-lg gap-2 text-[10px] font-bold uppercase tracking-widest data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:border-0 border border-transparent transition-all">
                    <Upload className="w-3.5 h-3.5" /> File
                  </TabsTrigger>
                </TabsList>
              </Tabs>
            </div>

            <div className="flex flex-wrap items-center gap-6 justify-center lg:justify-end w-full lg:w-auto">
              <div className="flex items-center space-x-8 glass-panel px-8 py-4 rounded-2xl">
                <div className="flex flex-col gap-2">
                  <Label htmlFor="auto-zoom" className="text-[9px] font-bold uppercase tracking-widest text-muted-foreground">Cinematic Zoom</Label>
                  <Switch id="auto-zoom" checked={isAutoZoomEnabled} onCheckedChange={setIsAutoZoomEnabled} disabled={!isStreaming} className="data-[state=checked]:bg-primary" />
                </div>
                <div className="w-px h-8 bg-white/10" />
                <div className="flex flex-col gap-2">
                  <Label htmlFor="low-light" className="text-[9px] font-bold uppercase tracking-widest text-muted-foreground">Enhanced Feed</Label>
                  <Switch id="low-light" checked={isLowLight} onCheckedChange={setIsLowLight} disabled={!isStreaming} className="data-[state=checked]:bg-primary" />
                </div>
              </div>

              {!isStreaming ? (
                <Button onClick={sourceMode === 'camera' ? startCamera : () => fileInputRef.current?.click()} disabled={isLoading || isModelLoading} className="h-14 px-10 rounded-xl text-[10px] font-bold uppercase tracking-[0.2em] transition-all hover:scale-[1.02] active:scale-95 bg-primary hover:bg-primary/90 text-primary-foreground emerald-glow">
                  {sourceMode === 'camera' ? <Camera className="mr-3 h-4 w-4" /> : <Upload className="mr-3 h-4 w-4" />}
                  {sourceMode === 'camera' ? "Start Feed" : "Choose File"}
                </Button>
              ) : (
                <Button variant="destructive" onClick={stopStream} className="h-14 px-10 rounded-xl text-[10px] font-bold uppercase tracking-[0.2em] transition-all hover:scale-[1.02] active:scale-95 bg-destructive hover:bg-destructive/90 text-destructive-foreground border-none">
                  <CameraOff className="mr-3 h-4 w-4" /> Kill Link
                </Button>
              )}
            </div>
          </div>
          <input type="file" ref={fileInputRef} onChange={handleFileChange} accept="video/mp4,video/webm,video/quicktime,video/*" className="hidden" />
          {hasCameraPermission === false && (
            <Alert variant="destructive" className="bg-destructive/10 border-destructive/30 text-destructive rounded-2xl p-4">
              <AlertCircle className="h-4 w-4" />
              <AlertTitle className="font-bold text-[10px] uppercase tracking-widest mb-1">Access Denied</AlertTitle>
              <AlertDescription className="text-xs">Camera access is blocked by security policy.</AlertDescription>
            </Alert>
          )}
        </CardFooter>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {[
          { title: "True Instance Focus", desc: "Only the clicked person stays sharp; others blur. Uses tracking ID and pixel-level segmentation mask—smooth boundary, no box.", icon: <Target className="w-5 h-5" /> },
          { title: "Smooth Mask", desc: "Pixel-level mask with soft edges. One selected subject in focus; all other people and background blurred.", icon: <Maximize className="w-5 h-5" /> },
          { title: "Neural Logic", desc: "BodyPix multi-person segmentation + COCO-SSD. Instance tracking across frames.", icon: <Cpu className="w-5 h-5" /> }
        ].map((f, i) => (
          <div key={i} className={cn("glass-panel p-8 rounded-2xl border border-white/10 space-y-4 hover:border-primary/30 transition-all group animate-fade-in", i === 0 && "stagger-1", i === 1 && "stagger-2", i === 2 && "stagger-3")}>
            <div className="p-3 bg-primary/10 text-primary w-fit rounded-xl border border-primary/20 group-hover:bg-primary group-hover:text-primary-foreground transition-colors">
              {f.icon}
            </div>
            <h3 className="font-heading text-sm font-bold uppercase tracking-widest text-foreground">{f.title}</h3>
            <p className="text-xs text-muted-foreground leading-relaxed font-medium">{f.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
};