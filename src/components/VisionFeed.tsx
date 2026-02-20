"use client"

import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Camera, CameraOff, Sparkles, RefreshCw, AlertCircle, Box, Cpu, MousePointer2, XCircle } from 'lucide-react';
import { Card, CardContent, CardFooter, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { cn } from '@/lib/utils';
import { useToast } from '@/hooks/use-toast';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';

// TensorFlow imports
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

export const VisionFeed: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const predictionsRef = useRef<cocoSsd.DetectedObject[]>([]);
  
  const [isStreaming, setIsStreaming] = useState(false);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [hasCameraPermission, setHasCameraPermission] = useState<boolean | null>(null);
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [selectedObject, setSelectedObject] = useState<cocoSsd.DetectedObject | null>(null);
  
  const { toast } = useToast();

  // Load COCO-SSD model once on start
  useEffect(() => {
    const loadModel = async () => {
      try {
        setIsModelLoading(true);
        // Ensure TF backend is ready
        await tf.ready();
        const loadedModel = await cocoSsd.load({
          base: 'lite_mobilenet_v2' // Using lite version for better browser performance
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
    setSelectedObject(null);
    
    // Clear canvas
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
    
    // Calculate click coordinates relative to the canvas resolution
    const x = (e.clientX - rect.left) * (canvas.width / rect.width);
    const y = (e.clientY - rect.top) * (canvas.height / rect.height);

    // Find the first object that contains the clicked point
    const clickedObj = predictionsRef.current.find(prediction => {
      const [bboxX, bboxY, width, height] = prediction.bbox;
      return x >= bboxX && x <= bboxX + width && y >= bboxY && y <= bboxY + height;
    });

    if (clickedObj) {
      setSelectedObject(clickedObj);
      toast({
        title: `Selected: ${clickedObj.class}`,
        description: `Tracking this object at ${Math.round(clickedObj.score * 100)}% confidence.`,
      });
    } else {
      setSelectedObject(null);
    }
  };

  // Handle detection loop (every 500ms)
  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    if (isStreaming && model && videoRef.current) {
      intervalId = setInterval(async () => {
        if (videoRef.current && videoRef.current.readyState === 4) {
          try {
            const predictions = await model.detect(videoRef.current);
            predictionsRef.current = predictions;
            
            // Try to update selected object position if it exists in the new predictions
            if (selectedObject) {
              const matchingObj = predictions.find(p => p.class === selectedObject.class);
              if (matchingObj) {
                // Simple heuristic: if same class exists, update the selection reference
                // A better tracker would use IOU (Intersection Over Union)
                setSelectedObject(matchingObj);
              }
            }
          } catch (err) {
            console.error("Detection error:", err);
          }
        }
      }, 500);
    }

    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isStreaming, model, selectedObject]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);

  // Effect to handle canvas resizing and real-time visualization overlay
  useEffect(() => {
    let animationFrameId: number;

    const render = () => {
      if (!isStreaming || !canvasRef.current || !videoRef.current) return;
      
      const canvas = canvasRef.current;
      const video = videoRef.current;
      const ctx = canvas.getContext('2d');
      
      if (!ctx || video.videoWidth === 0) {
        animationFrameId = requestAnimationFrame(render);
        return;
      }

      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // 1. Draw Object Detections
      const predictions = predictionsRef.current;
      
      predictions.forEach(prediction => {
        const [x, y, width, height] = prediction.bbox;
        const score = Math.round(prediction.score * 100);
        
        // Determine if this object is the selected one
        const isSelected = selectedObject && 
          prediction.class === selectedObject.class && 
          Math.abs(prediction.bbox[0] - selectedObject.bbox[0]) < 50;

        // Dynamic styling based on selection
        const primaryColor = isSelected ? '#22c55e' : '#ef4444'; // Green for selected, Red for others
        const labelBg = isSelected ? 'rgba(21, 128, 61, 0.85)' : 'rgba(185, 28, 28, 0.85)';

        ctx.strokeStyle = primaryColor;
        ctx.lineWidth = isSelected ? 5 : 2;

        // Bounding Box
        ctx.strokeRect(x, y, width, height);

        // Label Background
        const labelText = `${prediction.class}${isSelected ? ' (SELECTED)' : ''} ${score}%`;
        ctx.font = `bold ${Math.max(14, canvas.width * 0.015)}px sans-serif`;
        const textWidth = ctx.measureText(labelText).width;
        
        ctx.fillStyle = labelBg;
        ctx.fillRect(x, y - (canvas.width * 0.03), textWidth + 10, canvas.width * 0.03);
        
        ctx.fillStyle = 'white';
        ctx.fillText(labelText, x + 5, y - (canvas.width * 0.008));
      });

      // UI Frame Overlay (faded lavender)
      ctx.strokeStyle = 'rgba(208, 188, 255, 0.3)';
      ctx.lineWidth = 2;
      const margin = canvas.width * 0.04;
      const corner = canvas.width * 0.08;
      // Top Left
      ctx.beginPath(); ctx.moveTo(margin, margin + corner); ctx.lineTo(margin, margin); ctx.lineTo(margin + corner, margin); ctx.stroke();
      // Top Right
      ctx.beginPath(); ctx.moveTo(canvas.width - margin - corner, margin); ctx.lineTo(canvas.width - margin, margin); ctx.lineTo(canvas.width - margin, margin + corner); ctx.stroke();
      // Bottom Right
      ctx.beginPath(); ctx.moveTo(canvas.width - margin, canvas.height - margin - corner); ctx.lineTo(canvas.width - margin, canvas.height - margin); ctx.lineTo(canvas.width - margin - corner, canvas.height - margin); ctx.stroke();
      // Bottom Left
      ctx.beginPath(); ctx.moveTo(margin + corner, canvas.height - margin); ctx.lineTo(margin, canvas.height - margin); ctx.lineTo(margin, canvas.height - margin - corner); ctx.stroke();

      animationFrameId = requestAnimationFrame(render);
    };

    if (isStreaming) {
      animationFrameId = requestAnimationFrame(render);
    }

    return () => {
      cancelAnimationFrame(animationFrameId);
    };
  }, [isStreaming, selectedObject]);

  return (
    <div className="w-full max-w-4xl mx-auto p-4 md:p-8 space-y-8 animate-fade-in">
      <Card className="overflow-hidden shadow-2xl border-none bg-white/80 backdrop-blur-sm">
        <CardHeader className="text-center pb-4 relative">
          {selectedObject && (
            <div className="absolute top-4 right-4 animate-scale-in">
              <Badge variant="default" className="bg-green-600 hover:bg-green-700 gap-2 pl-3 py-1 pr-1">
                Selected: {selectedObject.class}
                <Button 
                  variant="ghost" 
                  size="icon" 
                  className="h-5 w-5 rounded-full hover:bg-green-500/50 p-0"
                  onClick={() => setSelectedObject(null)}
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
                <Cpu className="w-3 h-3" /> Neural Engine Active
              </Badge>
            )}
          </div>
          <CardTitle className="text-4xl font-headline font-bold text-primary flex items-center justify-center gap-3">
            <Sparkles className="w-8 h-8" />
            Vision Stream
          </CardTitle>
          <CardDescription className="text-lg">
            Click objects on the stream to select and track them.
          </CardDescription>
        </CardHeader>
        
        <CardContent className="p-0 relative">
          <div className={cn(
            "relative aspect-video bg-muted flex flex-col items-center justify-center overflow-hidden transition-all duration-500",
            !isStreaming && "bg-slate-100"
          )}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={cn(
                "w-full h-full object-cover transition-opacity duration-700 absolute inset-0",
                isStreaming ? "opacity-100" : "opacity-0"
              )}
            />
            
            <canvas
              ref={canvasRef}
              onClick={handleCanvasClick}
              className={cn(
                "absolute inset-0 z-10 transition-opacity duration-700 w-full h-full object-contain cursor-crosshair",
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

        <CardFooter className="flex flex-col sm:flex-row gap-4 p-8 justify-center bg-secondary/10 border-t border-secondary/20">
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
        </CardFooter>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 animate-fade-in delay-200">
        {[
          { title: "Point & Select", desc: "Click any detected object to highlight it globally.", icon: <MousePointer2 className="w-6 h-6 text-primary" /> },
          { title: "Visual Logic", desc: "Selected objects turn green, others stay red for clarity.", icon: <Box className="w-6 h-6 text-primary" /> },
          { title: "Neural Engine", desc: "Local TensorFlow COCO-SSD detection running at 2Hz.", icon: <Cpu className="w-6 h-6 text-primary" /> }
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