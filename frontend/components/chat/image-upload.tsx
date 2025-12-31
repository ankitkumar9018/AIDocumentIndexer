"use client";

import { useState, useRef, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { ImagePlus, X, Loader2, Image as ImageIcon } from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "sonner";

interface ImageAttachment {
  id: string;
  data: string; // Base64 encoded
  mimeType: string;
  name: string;
  preview: string; // Data URL for preview
}

interface ImageUploadProps {
  onImagesChange: (images: ImageAttachment[]) => void;
  images: ImageAttachment[];
  disabled?: boolean;
  maxImages?: number;
  maxSizeBytes?: number;
  className?: string;
}

const ACCEPTED_TYPES = ["image/jpeg", "image/png", "image/gif", "image/webp"];
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB default

export function ImageUpload({
  onImagesChange,
  images,
  disabled = false,
  maxImages = 4,
  maxSizeBytes = MAX_FILE_SIZE,
  className,
}: ImageUploadProps) {
  const [isProcessing, setIsProcessing] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const processFile = useCallback(
    async (file: File): Promise<ImageAttachment | null> => {
      // Validate file type
      if (!ACCEPTED_TYPES.includes(file.type)) {
        toast.error("Invalid file type", {
          description: `Only ${ACCEPTED_TYPES.map((t) => t.split("/")[1]).join(", ")} images are supported.`,
        });
        return null;
      }

      // Validate file size
      if (file.size > maxSizeBytes) {
        toast.error("File too large", {
          description: `Maximum file size is ${Math.round(maxSizeBytes / 1024 / 1024)}MB.`,
        });
        return null;
      }

      return new Promise((resolve) => {
        const reader = new FileReader();
        reader.onload = () => {
          const dataUrl = reader.result as string;
          // Extract base64 data (remove data URL prefix)
          const base64Data = dataUrl.split(",")[1];

          resolve({
            id: `img-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
            data: base64Data,
            mimeType: file.type,
            name: file.name,
            preview: dataUrl,
          });
        };
        reader.onerror = () => {
          toast.error("Failed to read file", {
            description: "Please try uploading the image again.",
          });
          resolve(null);
        };
        reader.readAsDataURL(file);
      });
    },
    [maxSizeBytes]
  );

  const handleFileSelect = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const files = event.target.files;
      if (!files || files.length === 0) return;

      setIsProcessing(true);

      try {
        const remainingSlots = maxImages - images.length;
        const filesToProcess = Array.from(files).slice(0, remainingSlots);

        if (files.length > remainingSlots) {
          toast.warning("Too many images", {
            description: `Only ${remainingSlots} more image(s) can be added. Maximum is ${maxImages}.`,
          });
        }

        const newImages: ImageAttachment[] = [];
        for (const file of filesToProcess) {
          const processed = await processFile(file);
          if (processed) {
            newImages.push(processed);
          }
        }

        if (newImages.length > 0) {
          onImagesChange([...images, ...newImages]);
        }
      } finally {
        setIsProcessing(false);
        // Reset input
        if (fileInputRef.current) {
          fileInputRef.current.value = "";
        }
      }
    },
    [images, maxImages, onImagesChange, processFile]
  );

  const handleRemove = useCallback(
    (id: string) => {
      onImagesChange(images.filter((img) => img.id !== id));
    },
    [images, onImagesChange]
  );

  const handlePaste = useCallback(
    async (event: React.ClipboardEvent) => {
      const items = event.clipboardData?.items;
      if (!items) return;

      const imageItems = Array.from(items).filter((item) =>
        item.type.startsWith("image/")
      );

      if (imageItems.length === 0) return;

      event.preventDefault();
      setIsProcessing(true);

      try {
        const remainingSlots = maxImages - images.length;
        const itemsToProcess = imageItems.slice(0, remainingSlots);

        const newImages: ImageAttachment[] = [];
        for (const item of itemsToProcess) {
          const file = item.getAsFile();
          if (file) {
            const processed = await processFile(file);
            if (processed) {
              newImages.push(processed);
            }
          }
        }

        if (newImages.length > 0) {
          onImagesChange([...images, ...newImages]);
          toast.success("Image pasted", {
            description: `${newImages.length} image(s) added.`,
          });
        }
      } finally {
        setIsProcessing(false);
      }
    },
    [images, maxImages, onImagesChange, processFile]
  );

  const canAddMore = images.length < maxImages;

  return (
    <div className={cn("flex flex-col gap-2", className)} onPaste={handlePaste}>
      {/* Image Previews */}
      {images.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {images.map((img) => (
            <div
              key={img.id}
              className="relative group w-16 h-16 rounded-lg overflow-hidden border bg-muted"
            >
              <img
                src={img.preview}
                alt={img.name}
                className="w-full h-full object-cover"
              />
              <Button
                variant="destructive"
                size="icon"
                className="absolute top-0.5 right-0.5 h-5 w-5 opacity-0 group-hover:opacity-100 transition-opacity"
                onClick={() => handleRemove(img.id)}
                disabled={disabled}
              >
                <X className="h-3 w-3" />
              </Button>
              <div className="absolute bottom-0 left-0 right-0 bg-black/50 text-white text-[10px] truncate px-1">
                {img.name}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Upload Button */}
      <input
        ref={fileInputRef}
        type="file"
        accept={ACCEPTED_TYPES.join(",")}
        multiple
        onChange={handleFileSelect}
        className="hidden"
        disabled={disabled || !canAddMore}
      />

      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={() => fileInputRef.current?.click()}
        disabled={disabled || isProcessing || !canAddMore}
        className="gap-2"
        title={
          canAddMore
            ? "Attach images (paste also works)"
            : `Maximum ${maxImages} images reached`
        }
      >
        {isProcessing ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <ImagePlus className="h-4 w-4" />
        )}
        <span className="hidden sm:inline">
          {images.length === 0 ? "Add Image" : `${images.length}/${maxImages}`}
        </span>
      </Button>
    </div>
  );
}

// Compact version for inline use in chat input
export function ImageUploadCompact({
  onImagesChange,
  images,
  disabled = false,
  maxImages = 4,
  className,
}: Omit<ImageUploadProps, "maxSizeBytes">) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isProcessing, setIsProcessing] = useState(false);

  const processFile = async (file: File): Promise<ImageAttachment | null> => {
    if (!ACCEPTED_TYPES.includes(file.type)) {
      toast.error("Invalid image type");
      return null;
    }
    if (file.size > MAX_FILE_SIZE) {
      toast.error("Image too large (max 10MB)");
      return null;
    }

    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = () => {
        const dataUrl = reader.result as string;
        resolve({
          id: `img-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`,
          data: dataUrl.split(",")[1],
          mimeType: file.type,
          name: file.name,
          preview: dataUrl,
        });
      };
      reader.onerror = () => resolve(null);
      reader.readAsDataURL(file);
    });
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files) return;

    setIsProcessing(true);
    const remainingSlots = maxImages - images.length;
    const filesToProcess = Array.from(files).slice(0, remainingSlots);

    const newImages: ImageAttachment[] = [];
    for (const file of filesToProcess) {
      const processed = await processFile(file);
      if (processed) newImages.push(processed);
    }

    if (newImages.length > 0) {
      onImagesChange([...images, ...newImages]);
    }

    setIsProcessing(false);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const canAddMore = images.length < maxImages;

  return (
    <>
      <input
        ref={fileInputRef}
        type="file"
        accept={ACCEPTED_TYPES.join(",")}
        multiple
        onChange={handleFileSelect}
        className="hidden"
        disabled={disabled || !canAddMore}
      />
      <Button
        type="button"
        variant={images.length > 0 ? "default" : "outline"}
        size="icon"
        onClick={() => fileInputRef.current?.click()}
        disabled={disabled || isProcessing || !canAddMore}
        className={cn("shrink-0", className)}
        title={
          canAddMore
            ? `Attach image (${images.length}/${maxImages})`
            : "Maximum images reached"
        }
      >
        {isProcessing ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : images.length > 0 ? (
          <div className="relative">
            <ImageIcon className="h-4 w-4" />
            <span className="absolute -top-1 -right-1 bg-primary-foreground text-primary text-[10px] rounded-full w-3.5 h-3.5 flex items-center justify-center">
              {images.length}
            </span>
          </div>
        ) : (
          <ImagePlus className="h-4 w-4" />
        )}
      </Button>
    </>
  );
}

// Preview bar component for showing attached images
export function ImagePreviewBar({
  images,
  onRemove,
  disabled = false,
  className,
}: {
  images: ImageAttachment[];
  onRemove: (id: string) => void;
  disabled?: boolean;
  className?: string;
}) {
  if (images.length === 0) return null;

  return (
    <div className={cn("flex gap-2 p-2 bg-muted/50 rounded-lg", className)}>
      {images.map((img) => (
        <div
          key={img.id}
          className="relative group w-12 h-12 rounded overflow-hidden border bg-background"
        >
          <img
            src={img.preview}
            alt={img.name}
            className="w-full h-full object-cover"
          />
          <Button
            variant="destructive"
            size="icon"
            className="absolute top-0 right-0 h-4 w-4 rounded-bl rounded-tr opacity-0 group-hover:opacity-100 transition-opacity"
            onClick={() => onRemove(img.id)}
            disabled={disabled}
          >
            <X className="h-2.5 w-2.5" />
          </Button>
        </div>
      ))}
      <span className="text-xs text-muted-foreground self-center ml-1">
        {images.length} image{images.length !== 1 ? "s" : ""} attached
      </span>
    </div>
  );
}
