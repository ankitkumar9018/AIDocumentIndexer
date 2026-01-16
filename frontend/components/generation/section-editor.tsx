'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import {
  CheckCircle2,
  Edit3,
  Loader2,
  Type,
  AlignLeft,
} from 'lucide-react';
import type { DocumentSectionReview, ContentReviewAction } from '@/lib/api';

interface SectionEditorProps {
  section: DocumentSectionReview;
  onApprove: () => void;
  onAction: (action: ContentReviewAction, feedback?: string) => void;
  isApproving: boolean;
}

export function SectionEditor({
  section,
  onApprove,
  onAction,
  isApproving,
}: SectionEditorProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editedSection, setEditedSection] = useState(section);
  const [isSaving, setIsSaving] = useState(false);
  const [feedbackDialog, setFeedbackDialog] = useState<{
    open: boolean;
    action: ContentReviewAction | null;
  }>({ open: false, action: null });
  const [feedbackText, setFeedbackText] = useState('');

  const handleSave = async () => {
    setIsSaving(true);
    try {
      // In a real implementation, this would call the API to update the section
      onAction('edit', JSON.stringify({
        title: editedSection.title,
        content: editedSection.content,
      }));
      setIsEditing(false);
    } catch (error) {
      console.error('Failed to save:', error);
    } finally {
      setIsSaving(false);
    }
  };

  const handleCancel = () => {
    setEditedSection(section);
    setIsEditing(false);
  };

  const handleActionWithFeedback = (action: ContentReviewAction) => {
    if (action === 'change_tone' || action === 'regenerate') {
      setFeedbackDialog({ open: true, action });
    } else {
      onAction(action);
    }
  };

  const handleSubmitFeedback = () => {
    if (feedbackDialog.action) {
      onAction(feedbackDialog.action, feedbackText);
    }
    setFeedbackDialog({ open: false, action: null });
    setFeedbackText('');
  };

  const getHeadingLevel = (level: number) => {
    switch (level) {
      case 1:
        return 'H1 - Main Section';
      case 2:
        return 'H2 - Subsection';
      case 3:
        return 'H3 - Sub-subsection';
      default:
        return `H${level}`;
    }
  };

  const statusColor = section.status === 'approved' ? 'bg-green-500' :
                      section.status === 'rejected' ? 'bg-red-500' :
                      'bg-yellow-500';

  const wordCount = section.content.split(/\s+/).filter(Boolean).length;
  const charCount = section.content.length;

  return (
    <div className="space-y-4">
      {/* Section Header */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Badge variant="outline" className="text-xs">
                Section {section.section_number}
              </Badge>
              <Badge variant="secondary" className="text-xs">
                {getHeadingLevel(section.level)}
              </Badge>
              <div className={`w-2 h-2 rounded-full ${statusColor}`} />
              <span className="text-xs text-muted-foreground capitalize">{section.status}</span>
            </div>
            <div className="flex items-center gap-2">
              {!isEditing ? (
                <Button variant="outline" size="sm" onClick={() => setIsEditing(true)}>
                  <Edit3 className="h-4 w-4 mr-1" />
                  Edit
                </Button>
              ) : (
                <>
                  <Button variant="ghost" size="sm" onClick={handleCancel}>
                    Cancel
                  </Button>
                  <Button size="sm" onClick={handleSave} disabled={isSaving}>
                    {isSaving ? (
                      <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                    ) : (
                      <CheckCircle2 className="h-4 w-4 mr-1" />
                    )}
                    Save
                  </Button>
                </>
              )}
            </div>
          </div>
        </CardHeader>
      </Card>

      {/* Section Content */}
      <Card>
        <CardContent className="pt-6 space-y-6">
          {/* Title */}
          <div className="space-y-2">
            <Label htmlFor="title" className="flex items-center gap-2">
              <Type className="h-4 w-4" />
              Section Title
            </Label>
            {isEditing ? (
              <Input
                id="title"
                value={editedSection.title}
                onChange={(e) => setEditedSection({ ...editedSection, title: e.target.value })}
                placeholder="Section title"
                className={`font-semibold ${
                  section.level === 1 ? 'text-xl' :
                  section.level === 2 ? 'text-lg' :
                  'text-base'
                }`}
              />
            ) : (
              <h3 className={`font-semibold ${
                section.level === 1 ? 'text-xl' :
                section.level === 2 ? 'text-lg' :
                'text-base'
              }`}>
                {section.title}
              </h3>
            )}
          </div>

          <Separator />

          {/* Content */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <Label htmlFor="content" className="flex items-center gap-2">
                <AlignLeft className="h-4 w-4" />
                Content
              </Label>
              <span className="text-xs text-muted-foreground">
                {wordCount} words • {charCount} characters
              </span>
            </div>
            {isEditing ? (
              <Textarea
                id="content"
                value={editedSection.content}
                onChange={(e) => setEditedSection({ ...editedSection, content: e.target.value })}
                placeholder="Section content..."
                rows={12}
                className="font-normal leading-relaxed"
              />
            ) : (
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <p className="whitespace-pre-wrap leading-relaxed">{section.content}</p>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Content Stats */}
      <Card>
        <CardContent className="pt-4">
          <div className="grid grid-cols-3 gap-4 text-center">
            <div>
              <p className="text-2xl font-bold text-primary">{wordCount}</p>
              <p className="text-xs text-muted-foreground">Words</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-primary">{charCount}</p>
              <p className="text-xs text-muted-foreground">Characters</p>
            </div>
            <div>
              <p className="text-2xl font-bold text-primary">
                {Math.ceil(wordCount / 200)}
              </p>
              <p className="text-xs text-muted-foreground">Min Read Time</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Feedback Dialog */}
      <Dialog open={feedbackDialog.open} onOpenChange={(open) => setFeedbackDialog({ ...feedbackDialog, open })}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {feedbackDialog.action === 'change_tone' ? 'Change Tone' : 'Regenerate Content'}
            </DialogTitle>
            <DialogDescription>
              {feedbackDialog.action === 'change_tone'
                ? 'Describe how you want to change the tone of this section.'
                : 'Provide instructions for regenerating this section.'}
            </DialogDescription>
          </DialogHeader>
          <Textarea
            value={feedbackText}
            onChange={(e) => setFeedbackText(e.target.value)}
            placeholder={
              feedbackDialog.action === 'change_tone'
                ? 'e.g., Make it more technical, more accessible, more persuasive...'
                : 'e.g., Add more examples, focus on specific aspects, expand on key points...'
            }
            rows={4}
          />
          <DialogFooter>
            <Button variant="outline" onClick={() => setFeedbackDialog({ open: false, action: null })}>
              Cancel
            </Button>
            <Button onClick={handleSubmitFeedback} disabled={!feedbackText.trim()}>
              Submit
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
