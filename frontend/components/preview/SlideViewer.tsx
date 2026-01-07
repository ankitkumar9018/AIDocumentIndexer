'use client';

import { useState } from 'react';
import { ChevronLeft, ChevronRight, Loader2, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';
import { usePreviewSlides } from '@/lib/api';

interface SlideViewerProps {
  jobId: string;
  className?: string;
}

export function SlideViewer({ jobId, className }: SlideViewerProps) {
  const [currentSlide, setCurrentSlide] = useState(0);
  const { data, isLoading, error } = usePreviewSlides(jobId);

  if (isLoading) {
    return (
      <div className={cn('flex items-center justify-center h-[400px] bg-muted rounded-lg', className)}>
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  if (error) {
    return (
      <div className={cn('flex flex-col items-center justify-center h-[400px] bg-muted rounded-lg gap-2', className)}>
        <AlertCircle className="h-8 w-8 text-destructive" />
        <p className="text-sm text-muted-foreground">Failed to load slides</p>
      </div>
    );
  }

  if (!data || data.slides.length === 0) {
    return (
      <div className={cn('flex items-center justify-center h-[400px] bg-muted rounded-lg', className)}>
        <p className="text-sm text-muted-foreground">No slides available</p>
      </div>
    );
  }

  const slides = data.slides;
  const totalSlides = slides.length;

  const goToPrevious = () => {
    setCurrentSlide((prev) => (prev > 0 ? prev - 1 : prev));
  };

  const goToNext = () => {
    setCurrentSlide((prev) => (prev < totalSlides - 1 ? prev + 1 : prev));
  };

  return (
    <div className={cn('flex flex-col gap-4', className)}>
      {/* Main slide viewer */}
      <div className="relative bg-black rounded-lg overflow-hidden">
        <img
          src={`data:image/png;base64,${slides[currentSlide]}`}
          alt={`Slide ${currentSlide + 1}`}
          className="w-full h-auto"
        />

        {/* Navigation overlay */}
        <div className="absolute inset-0 flex items-center justify-between px-2">
          <Button
            variant="ghost"
            size="icon"
            className="bg-black/50 hover:bg-black/70 text-white rounded-full"
            onClick={goToPrevious}
            disabled={currentSlide === 0}
          >
            <ChevronLeft className="h-6 w-6" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="bg-black/50 hover:bg-black/70 text-white rounded-full"
            onClick={goToNext}
            disabled={currentSlide === totalSlides - 1}
          >
            <ChevronRight className="h-6 w-6" />
          </Button>
        </div>

        {/* Slide counter */}
        <div className="absolute bottom-2 left-1/2 -translate-x-1/2 bg-black/50 text-white text-sm px-3 py-1 rounded-full">
          {currentSlide + 1} / {totalSlides}
        </div>
      </div>

      {/* Thumbnail strip */}
      <div className="flex gap-2 overflow-x-auto pb-2">
        {slides.map((slide, index) => (
          <button
            key={index}
            onClick={() => setCurrentSlide(index)}
            className={cn(
              'flex-shrink-0 w-20 h-12 rounded border-2 overflow-hidden transition-all',
              currentSlide === index
                ? 'border-primary ring-2 ring-primary/20'
                : 'border-muted-foreground/20 hover:border-muted-foreground/40'
            )}
          >
            <img
              src={`data:image/png;base64,${slide}`}
              alt={`Slide ${index + 1} thumbnail`}
              className="w-full h-full object-cover"
            />
          </button>
        ))}
      </div>
    </div>
  );
}
