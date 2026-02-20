"use client"

import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Camera, CameraOff, Sparkles, RefreshCw, AlertCircle, Box, Cpu, MousePointer2, XCircle, Moon, Sun } from 'lucide-react';
import { Card, CardContent, CardFooter, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import { useToast } from '@/hooks/use-toast';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Switch } from '@/components/ui/switch';
import { Label } from '@/components/ui/label';

// TensorFlow imports
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

/**
 * Calculates Intersection over Union (IoU) between two bounding boxes.
 * Bounding boxes are expected in [x, y, width, height] format.
 */
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

export const VisionFeed: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // Refs for tracking and detection state without triggering re-renders
  const predictionsRef = useRef<cocoSsd.DetectedObject[]>([]);
  const frameCountRef = useRef(0);
  const isDetectingRef = useRef(false);
  const selectedObjectRef = useRef<cocoSsd.DetectedObject | null>(null);
  
  const [isStreaming, setIsStreaming] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [hasCameraPermission, setHasCameraPermission] = useState<boolean | null>(null);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [isLowLight, setIsLowLight] = useState(false);
  
  // Keep a state version for UI labels, synchronized with ref
  const [selectedLabel, setSelectedLabel] = useState<string | null>(null);
  
  const { toast } = useToast();

  // Load COCO-SSD model once on start
  useEffect(() => {
    const loadModel = async () => {
      try {
        setIsModelLoading(true);
        await tf.ready();
        const loadedModel = await cocoSsd.load({
          base: 'lite_mobilenet_v2'
        });
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

  const startCamera = async () => {
    setIsLoading(true);
    setHasCameraPermission(null);
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
      setHasCameraPermission(true);
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        await videoRef.current.play();
      }
      setIsStreaming(true);
    } catch (err) {
      console.error("Error accessing camera:", err);
      setHasCameraPermission(false);
      toast({
        title: "Camera Access Failed",
        description: "Please enable camera permissions in your browser settings.",
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
    predictionsRef.current = [];
    selectedObjectRef.current = null;
    setSelectedLabel(null);
    
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx?.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, [stream]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!isStreaming || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    // Convert click coordinates to canvas internal resolution
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);

    // Find all objects containing the click point
    const candidates = predictionsRef.current.filter(prediction => {
      const [bboxX, bboxY, width, height] = prediction.bbox;
      return x >= bboxX && x <= bboxX + width && y >= bboxY && y <= bboxY + height;
    });

    if (candidates.length > 0) {
      // Pick the smallest box if multiple overlap (most precise target)
      const clickedObj = candidates.sort((a, b) => (a.bbox[2] * a.bbox[3]) - (b.bbox[2] * b.bbox[3]))[0];
      
      // Update ref immediately to reset tracking logic to the new object
      // Clone the object to prevent reference updates from standard detection cycles affecting the switch logic
      selectedObjectRef.current = JSON.parse(JSON.stringify(clickedObj));
      setSelectedLabel(clickedObj.class);
      
      toast({
        title: `Focus Locked: ${clickedObj.class}`,
        description: `Tracking target with ${Math.round(clickedObj.score * 100)}% confidence.`,
      });
    } else {
      selectedObjectRef.current = null;
      setSelectedLabel(null);
    }
  };

  // Unified Visualization and Detection Effect
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

      // Sync canvas resolution
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      // 1. Detection Throttling (Every 10 frames)
      if (frameCountRef.current % 10 === 0 && model && !isDetectingRef.current) {
        isDetectingRef.current = true;
        
        // Inference runs on the raw video stream
        model.detect(video).then(predictions => {
          predictionsRef.current = predictions;
          
          // IoU Tracking Update
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

            // Update focus if match is found, or clear if object is completely lost
            if (maxIoU > 0.4 && bestMatch) {
              selectedObjectRef.current = bestMatch;
            } else if (maxIoU < 0.1) {
              // Only clear automatically if it's truly gone to avoid flicker
              selectedObjectRef.current = null;
              setSelectedLabel(null);
            }
          }
          
          isDetectingRef.current = false;
        }).catch(err => {
          console.error("AI Detection error:", err);
          isDetectingRef.current = false;
        });
      }
      frameCountRef.current++;

      // Enhancement filter setup
      const enhancementFilter = isLowLight ? "contrast(1.2) brightness(1.1) " : "";

      // 2. Draw Blurred Background
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.save();
      // Combine background blur with optional low-light enhancement
      ctx.filter = `${enhancementFilter}blur(12px) brightness(0.8)`;
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      ctx.restore();

      // 3. Draw Sharp Focus Area
      const activeSelection = selectedObjectRef.current;
      if (activeSelection) {
        const [x, y, width, height] = activeSelection.bbox;
        ctx.save();
        ctx.filter = isLowLight ? "contrast(1.2) brightness(1.1)" : "none";
        ctx.beginPath();
        ctx.rect(x, y, width, height);
        ctx.clip();
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        ctx.restore();
      }

      // 4. Draw Overlays
      const predictions = predictionsRef.current;
      predictions.forEach(prediction => {
        const [x, y, width, height] = prediction.bbox;
        const score = Math.round(prediction.score * 100);
        
        // Check if this box is the current selected object
        const isSelected = activeSelection && 
          prediction.class === activeSelection.class && 
          calculateIoU(prediction.bbox, activeSelection.bbox) > 0.8;

        const primaryColor = isSelected ? '#22c55e' : 'rgba(239, 68, 68, 0.4)';
        const labelBg = isSelected ? 'rgba(21, 128, 61, 0.9)' : 'rgba(185, 28, 28, 0.5)';

        ctx.strokeStyle = primaryColor;
        ctx.lineWidth = isSelected ? 4 : 2;
        ctx.strokeRect(x, y, width, height);

        const labelText = `${prediction.class} ${score}%`;
        ctx.font = `bold ${Math.max(14, canvas.width * 0.012)}px sans-serif`;
        const textWidth = ctx.measureText(labelText).width;
        
        ctx.fillStyle = labelBg;
        ctx.fillRect(x, y - (canvas.width * 0.03), textWidth + 12, canvas.width * 0.03);
        
        ctx.fillStyle = 'white';
        ctx.fillText(labelText, x + 6, y - (canvas.width * 0.008));
      });

      // 5. Draw HUD Corners
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
      ctx.lineWidth = 2;
      const m = canvas.width * 0.05;
      const l = canvas.width * 0.08;
      ctx.beginPath(); ctx.moveTo(m, m + l); ctx.lineTo(m, m); ctx.lineTo(m + l, m); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(canvas.width - m - l, m); ctx.lineTo(canvas.width - m, m); ctx.lineTo(canvas.width - m, m + l); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(canvas.width - m, canvas.height - m - l); ctx.lineTo(canvas.width - m, canvas.height - m); ctx.lineTo(canvas.width - m - l, canvas.height - m); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(m + l, canvas.height - m); ctx.lineTo(m, canvas.height - m); ctx.lineTo(m, canvas.height - m - l); ctx.stroke();

      animationFrameId = requestAnimationFrame(render);
    };

    if (isStreaming) {
      animationFrameId = requestAnimationFrame(render);
    }

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [isStreaming, model, isLowLight]);

  return (
    <div className="w-full max-w-4xl mx-auto p-4 md:p-8 space-y-8 animate-fade-in">
      <Card className="overflow-hidden shadow-2xl border-none bg-white/80 backdrop-blur-sm">
        <CardHeader className="text-center pb-4 relative">
          {selectedLabel && (
            <div className="absolute top-4 right-4 animate-scale-in">
              <Badge variant="default" className="bg-green-600 hover:bg-green-700 gap-2 pl-3 py-1 pr-1 shadow-lg">
                Focus: {selectedLabel}
                <Button 
                  variant="ghost" 
                  size="icon" 
                  className="h-5 w-5 rounded-full hover:bg-green-500/50 p-0"
                  onClick={() => {
                    selectedObjectRef.current = null;
                    setSelectedLabel(null);
                  }}
                >
                  <XCircle className="w-3.5 h-3.5" />
                </Button>
              </Badge>
            </div>
          )}
          <div className="flex justify-center mb-2">
            {isModelLoading ? (
              <Badge variant="secondary" className="animate-pulse gap-1">
                <RefreshCw className="w-3 h-3 animate-spin" /> Initializing AI...
              </Badge>
            ) : (
              <Badge variant="outline" className="text-primary border-primary/20 gap-1 bg-primary/5">
                <Cpu className="w-3 h-3" /> Throttled Inference Loop
              </Badge>
            )}
          </div>
          <CardTitle className="text-4xl font-headline font-bold text-primary flex items-center justify-center gap-3">
            <Sparkles className="w-8 h-8" />
            Vision Stream
          </CardTitle>
          <CardDescription className="text-lg">
            Selective background blur with real-time AI object tracking.
          </CardDescription>
        </CardHeader>
        
        <CardContent className="p-0 relative">
          <div className={cn(
            "relative aspect-video bg-muted flex flex-col items-center justify-center overflow-hidden transition-all duration-500",
            !isStreaming && "bg-slate-100"
          )}>
            {/* Using opacity-0 instead of display:none to keep the video element active for the render loop */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="absolute inset-0 w-full h-full object-cover opacity-0 pointer-events-none"
            />
            
            <canvas
              ref={canvasRef}
              onClick={handleCanvasClick}
              className={cn(
                "absolute inset-0 z-10 transition-opacity duration-700 w-full h-full object-cover cursor-crosshair",
                isStreaming ? "opacity-100" : "opacity-0 pointer-events-none"
              )}
            />

            {!isStreaming && !isLoading && (
              <div className="z-10 flex flex-col items-center gap-4 text-muted-foreground animate-scale-in">
                <div className="p-8 rounded-full bg-secondary/50">
                  <Camera className="w-16 h-16 text-primary" />
                </div>
                <p className="font-medium">Stream is currently offline</p>
              </div>
            )}

            {isLoading && (
              <div className="absolute inset-0 bg-white/40 z-20 flex items-center justify-center backdrop-blur-[2px]">
                <RefreshCw className="w-12 h-12 text-primary animate-spin" />
              </div>
            )}
          </div>

          {hasCameraPermission === false && (
            <div className="px-6 py-4">
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Access Required</AlertTitle>
                <AlertDescription>
                  Camera access was denied. Please check your browser's site permissions.
                </AlertDescription>
              </Alert>
            </div>
          )}
        </CardContent>

        <CardFooter className="flex flex-col gap-6 p-8 bg-secondary/10 border-t border-secondary/20">
          <div className="flex flex-col sm:flex-row items-center justify-between w-full gap-4">
            <div className="flex items-center space-x-4 bg-white/50 px-4 py-2 rounded-full border border-primary/10 shadow-sm">
              <div className="flex items-center gap-2">
                {isLowLight ? <Moon className="w-4 h-4 text-primary" /> : <Sun className="w-4 h-4 text-muted-foreground" />}
                <Label htmlFor="low-light" className="text-sm font-semibold whitespace-nowrap">
                  Low Light Mode
                </Label>
              </div>
              <Switch 
                id="low-light" 
                checked={isLowLight} 
                onCheckedChange={setIsLowLight}
                disabled={!isStreaming}
              />
            </div>

            <div className="flex gap-4">
              {!isStreaming ? (
                <Button 
                  size="lg" 
                  onClick={startCamera} 
                  disabled={isLoading || isModelLoading}
                  className="px-10 h-14 rounded-full text-lg font-semibold transition-all hover:scale-105 active:scale-95 shadow-lg shadow-primary/20"
                >
                  {isLoading ? (
                    <><RefreshCw className="mr-2 h-5 w-5 animate-spin" /> Initializing...</>
                  ) : (
                    <><Camera className="mr-2 h-5 w-5" /> Start Live Stream</>
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
            </div>
          </div>
          
          <div className="text-center">
            <p className="text-xs text-muted-foreground max-w-sm mx-auto">
              Click any object in the stream to lock focus. {isLowLight && "Low Light Mode is applying contrast enhancement filters."}
            </p>
          </div>
        </CardFooter>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 animate-fade-in delay-200">
        {[
          { title: "Optimized Loop", desc: "Detection runs every 10 frames while rendering stays at a smooth 60 FPS.", icon: <Cpu className="w-6 h-6 text-primary" /> },
          { title: "Smart Tracking", desc: "IoU-based matching follows your target even between inference cycles.", icon: <Box className="w-6 h-6 text-primary" /> },
          { title: "Interactive Focus", desc: "Click any detected object to instantly focus the AI's selective attention.", icon: <MousePointer2 className="w-6 h-6 text-primary" /> }
        ].map((feature, i) => (
          <Card key={i} className="bg-white/60 border-none shadow-sm hover:shadow-md transition-shadow">
            <CardHeader className="pb-2">
              <div className="mb-2">{feature.icon}</div>
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