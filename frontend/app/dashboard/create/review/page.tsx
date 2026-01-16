'use client';

import { useEffect, useState } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Loader2, ArrowLeft, AlertCircle } from 'lucide-react';
import { toast } from 'sonner';
import { api } from '@/lib/api';
import { ContentReviewPanel } from '@/components/generation';

export default function ContentReviewPage() {
  const router = useRouter();
  const searchParams = useSearchParams();

  const sessionId = searchParams.get('session');
  const contentType = searchParams.get('type') as 'pptx' | 'docx' | 'xlsx' | null;
  const jobId = searchParams.get('job');

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!sessionId || !contentType) {
      setError('Missing session ID or content type');
      setIsLoading(false);
      return;
    }

    // Verify session exists
    api.getContentReviewStatus(sessionId)
      .then(() => {
        setIsLoading(false);
      })
      .catch((err) => {
        console.error('Failed to load review session:', err);
        setError('Review session not found or expired');
        setIsLoading(false);
      });
  }, [sessionId, contentType]);

  const handleComplete = async (content: Record<string, unknown>) => {
    toast.success('Review completed! Generating document...');

    // Navigate back to create page with the approved content
    // The create page will use this to trigger final document generation
    if (jobId) {
      router.push(`/dashboard/create?job=${jobId}&reviewComplete=true`);
    } else {
      router.push('/dashboard/create');
    }
  };

  const handleCancel = () => {
    if (confirm('Are you sure you want to cancel the review? Your progress will be lost.')) {
      if (sessionId) {
        api.deleteContentReviewSession(sessionId).catch(console.error);
      }
      router.push('/dashboard/create');
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4 text-muted-foreground" />
          <p className="text-muted-foreground">Loading review session...</p>
        </div>
      </div>
    );
  }

  if (error || !sessionId || !contentType) {
    return (
      <div className="container mx-auto py-8">
        <Card className="max-w-md mx-auto">
          <CardHeader>
            <div className="flex items-center gap-2 text-destructive">
              <AlertCircle className="h-5 w-5" />
              <CardTitle>Error</CardTitle>
            </div>
            <CardDescription>
              {error || 'Invalid review session parameters'}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Button
              variant="outline"
              onClick={() => router.push('/dashboard/create')}
              className="w-full"
            >
              <ArrowLeft className="h-4 w-4 mr-2" />
              Back to Create
            </Button>
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div className="h-[calc(100vh-4rem)]">
      <ContentReviewPanel
        sessionId={sessionId}
        contentType={contentType}
        onComplete={handleComplete}
        onCancel={handleCancel}
      />
    </div>
  );
}
