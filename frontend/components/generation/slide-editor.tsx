'use client';

import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Textarea } from '@/components/ui/textarea';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog';
import {
  CheckCircle2,
  Edit3,
  Plus,
  Trash2,
  GripVertical,
  Image,
  MessageSquare,
  Loader2,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import type { SlideContentReview, ContentReviewAction } from '@/lib/api';

interface SlideEditorProps {
  slide: SlideContentReview;
  onUpdate: (updates: Partial<SlideContentReview>) => Promise<void>;
  onApprove: () => void;
  onAction: (action: ContentReviewAction, feedback?: string) => void;
  isApproving: boolean;
}

const LAYOUT_OPTIONS = [
  { value: 'title_only', label: 'Title Only' },
  { value: 'title_content', label: 'Title + Content' },
  { value: 'two_column', label: 'Two Columns' },
  { value: 'title_bullets', label: 'Title + Bullets' },
  { value: 'image_text', label: 'Image + Text' },
  { value: 'section_header', label: 'Section Header' },
];

export function SlideEditor({
  slide,
  onUpdate,
  onApprove,
  onAction,
  isApproving,
}: SlideEditorProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [editedSlide, setEditedSlide] = useState(slide);
  const [isSaving, setIsSaving] = useState(false);
  const [feedbackDialog, setFeedbackDialog] = useState<{
    open: boolean;
    action: ContentReviewAction | null;
  }>({ open: false, action: null });
  const [feedbackText, setFeedbackText] = useState('');
  const [showSpeakerNotes, setShowSpeakerNotes] = useState(false);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await onUpdate(editedSlide);
      setIsEditing(false);
    } catch (error) {
      console.error('Failed to save:', error);
    } finally {
      setIsSaving(false);
    }
  };

  const handleCancel = () => {
    setEditedSlide(slide);
    setIsEditing(false);
  };

  const handleBulletChange = (index: number, text: string) => {
    const newBullets = [...editedSlide.bullets];
    newBullets[index] = { ...newBullets[index], text };
    setEditedSlide({ ...editedSlide, bullets: newBullets });
  };

  const handleAddBullet = () => {
    setEditedSlide({
      ...editedSlide,
      bullets: [...editedSlide.bullets, { text: '', sub_bullets: [] }],
    });
  };

  const handleRemoveBullet = (index: number) => {
    const newBullets = editedSlide.bullets.filter((_, i) => i !== index);
    setEditedSlide({ ...editedSlide, bullets: newBullets });
  };

  const handleSubBulletChange = (bulletIndex: number, subIndex: number, text: string) => {
    const newBullets = [...editedSlide.bullets];
    const subBullets = [...(newBullets[bulletIndex].sub_bullets || [])];
    subBullets[subIndex] = text;
    newBullets[bulletIndex] = { ...newBullets[bulletIndex], sub_bullets: subBullets };
    setEditedSlide({ ...editedSlide, bullets: newBullets });
  };

  const handleAddSubBullet = (bulletIndex: number) => {
    const newBullets = [...editedSlide.bullets];
    const subBullets = [...(newBullets[bulletIndex].sub_bullets || []), ''];
    newBullets[bulletIndex] = { ...newBullets[bulletIndex], sub_bullets: subBullets };
    setEditedSlide({ ...editedSlide, bullets: newBullets });
  };

  const handleRemoveSubBullet = (bulletIndex: number, subIndex: number) => {
    const newBullets = [...editedSlide.bullets];
    const subBullets = (newBullets[bulletIndex].sub_bullets || []).filter((_, i) => i !== subIndex);
    newBullets[bulletIndex] = { ...newBullets[bulletIndex], sub_bullets: subBullets };
    setEditedSlide({ ...editedSlide, bullets: newBullets });
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

  // Handle both frontend and backend status values
  const statusColor = (slide.status === 'approved' || slide.status === 'final') ? 'bg-green-500' :
                      slide.status === 'rejected' ? 'bg-red-500' :
                      (slide.status === 'edited' || slide.status === 'regenerating') ? 'bg-blue-500' :
                      'bg-yellow-500'; // draft, pending, pending_review

  return (
    <div className="space-y-4">
      {/* Slide Header */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Badge variant="outline" className="text-xs">
                Slide {slide.slide_number}
              </Badge>
              <Badge variant="secondary" className="text-xs">
                {LAYOUT_OPTIONS.find(l => l.value === slide.layout)?.label || slide.layout}
              </Badge>
              <div className={`w-2 h-2 rounded-full ${statusColor}`} />
              <span className="text-xs text-muted-foreground capitalize">{slide.status}</span>
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

      {/* Slide Content */}
      <Card>
        <CardContent className="pt-6 space-y-6">
          {/* Title */}
          <div className="space-y-2">
            <Label htmlFor="title">Title</Label>
            {isEditing ? (
              <Input
                id="title"
                value={editedSlide.title}
                onChange={(e) => setEditedSlide({ ...editedSlide, title: e.target.value })}
                placeholder="Slide title"
                maxLength={80}
              />
            ) : (
              <p className="text-lg font-semibold">{slide.title}</p>
            )}
            {isEditing && (
              <p className="text-xs text-muted-foreground">
                {editedSlide.title.length}/80 characters
              </p>
            )}
          </div>

          {/* Subtitle */}
          {(slide.subtitle || isEditing) && (
            <div className="space-y-2">
              <Label htmlFor="subtitle">Subtitle</Label>
              {isEditing ? (
                <Input
                  id="subtitle"
                  value={editedSlide.subtitle || ''}
                  onChange={(e) => setEditedSlide({ ...editedSlide, subtitle: e.target.value })}
                  placeholder="Slide subtitle (optional)"
                  maxLength={120}
                />
              ) : (
                slide.subtitle && <p className="text-muted-foreground">{slide.subtitle}</p>
              )}
            </div>
          )}

          {/* Layout Selector (edit mode only) */}
          {isEditing && (
            <div className="space-y-2">
              <Label>Layout</Label>
              <Select
                value={editedSlide.layout}
                onValueChange={(value) => setEditedSlide({ ...editedSlide, layout: value })}
              >
                <SelectTrigger className="w-[200px]">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {LAYOUT_OPTIONS.map((option) => (
                    <SelectItem key={option.value} value={option.value}>
                      {option.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          <Separator />

          {/* Bullets */}
          {slide.bullets.length > 0 && (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label>Bullet Points</Label>
                {isEditing && (
                  <Button variant="ghost" size="sm" onClick={handleAddBullet}>
                    <Plus className="h-4 w-4 mr-1" />
                    Add Bullet
                  </Button>
                )}
              </div>
              <div className="space-y-3">
                {(isEditing ? editedSlide.bullets : slide.bullets).map((bullet, index) => (
                  <div key={index} className="space-y-2">
                    <div className="flex items-start gap-2">
                      {isEditing && (
                        <GripVertical className="h-5 w-5 text-muted-foreground mt-2 cursor-move" />
                      )}
                      <div className="flex-1">
                        {isEditing ? (
                          <div className="flex items-center gap-2">
                            <Input
                              value={bullet.text}
                              onChange={(e) => handleBulletChange(index, e.target.value)}
                              placeholder={`Bullet point ${index + 1}`}
                            />
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={() => handleRemoveBullet(index)}
                            >
                              <Trash2 className="h-4 w-4 text-destructive" />
                            </Button>
                          </div>
                        ) : (
                          <p className="text-sm flex items-start gap-2">
                            <span className="text-primary mt-1">•</span>
                            {bullet.text}
                          </p>
                        )}
                      </div>
                    </div>

                    {/* Sub-bullets */}
                    {bullet.sub_bullets && bullet.sub_bullets.length > 0 && (
                      <div className="ml-8 space-y-2">
                        {bullet.sub_bullets.map((subBullet, subIndex) => (
                          <div key={subIndex} className="flex items-center gap-2">
                            {isEditing ? (
                              <>
                                <Input
                                  value={subBullet}
                                  onChange={(e) => handleSubBulletChange(index, subIndex, e.target.value)}
                                  placeholder={`Sub-bullet ${subIndex + 1}`}
                                  className="flex-1"
                                />
                                <Button
                                  variant="ghost"
                                  size="icon"
                                  onClick={() => handleRemoveSubBullet(index, subIndex)}
                                >
                                  <Trash2 className="h-3 w-3 text-destructive" />
                                </Button>
                              </>
                            ) : (
                              <p className="text-sm text-muted-foreground flex items-start gap-2">
                                <span className="mt-1">◦</span>
                                {subBullet}
                              </p>
                            )}
                          </div>
                        ))}
                      </div>
                    )}

                    {isEditing && (
                      <Button
                        variant="ghost"
                        size="sm"
                        className="ml-8"
                        onClick={() => handleAddSubBullet(index)}
                      >
                        <Plus className="h-3 w-3 mr-1" />
                        Add Sub-bullet
                      </Button>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Body Text */}
          {(slide.body_text || isEditing) && (
            <>
              <Separator />
              <div className="space-y-2">
                <Label htmlFor="body_text">Body Text</Label>
                {isEditing ? (
                  <Textarea
                    id="body_text"
                    value={editedSlide.body_text || ''}
                    onChange={(e) => setEditedSlide({ ...editedSlide, body_text: e.target.value })}
                    placeholder="Additional body text (optional)"
                    rows={4}
                    maxLength={600}
                  />
                ) : (
                  slide.body_text && <p className="text-sm">{slide.body_text}</p>
                )}
                {isEditing && (
                  <p className="text-xs text-muted-foreground">
                    {(editedSlide.body_text || '').length}/600 characters
                  </p>
                )}
              </div>
            </>
          )}

          {/* Image Description */}
          {(slide.image_description || isEditing) && (
            <>
              <Separator />
              <div className="space-y-2">
                <Label htmlFor="image_description" className="flex items-center gap-2">
                  <Image className="h-4 w-4" />
                  Image Description
                </Label>
                {isEditing ? (
                  <Input
                    id="image_description"
                    value={editedSlide.image_description || ''}
                    onChange={(e) => setEditedSlide({ ...editedSlide, image_description: e.target.value })}
                    placeholder="Description of the image to generate"
                    maxLength={150}
                  />
                ) : (
                  slide.image_description && (
                    <p className="text-sm text-muted-foreground italic">
                      {slide.image_description}
                    </p>
                  )
                )}
              </div>
            </>
          )}
        </CardContent>
      </Card>

      {/* Speaker Notes */}
      <Card>
        <CardHeader className="pb-3 cursor-pointer" onClick={() => setShowSpeakerNotes(!showSpeakerNotes)}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <MessageSquare className="h-4 w-4" />
              <CardTitle className="text-sm">Speaker Notes</CardTitle>
            </div>
            {showSpeakerNotes ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
          </div>
        </CardHeader>
        {showSpeakerNotes && (
          <CardContent>
            {isEditing ? (
              <Textarea
                value={editedSlide.speaker_notes || ''}
                onChange={(e) => setEditedSlide({ ...editedSlide, speaker_notes: e.target.value })}
                placeholder="Notes for the presenter"
                rows={4}
                maxLength={500}
              />
            ) : (
              <p className="text-sm text-muted-foreground whitespace-pre-wrap">
                {slide.speaker_notes || 'No speaker notes'}
              </p>
            )}
            {isEditing && (
              <p className="text-xs text-muted-foreground mt-2">
                {(editedSlide.speaker_notes || '').length}/500 characters
              </p>
            )}
          </CardContent>
        )}
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
                ? 'Describe how you want to change the tone of this slide.'
                : 'Provide instructions for regenerating this slide.'}
            </DialogDescription>
          </DialogHeader>
          <Textarea
            value={feedbackText}
            onChange={(e) => setFeedbackText(e.target.value)}
            placeholder={
              feedbackDialog.action === 'change_tone'
                ? 'e.g., Make it more formal, more casual, more persuasive...'
                : 'e.g., Focus more on technical details, add examples...'
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
