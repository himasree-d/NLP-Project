#!/usr/bin/env python3
import click
from pathlib import Path
from .core import Transcriber

@click.command()
@click.argument('input_file', type=click.Path(exists=True))
@click.option('-o', '--output', help='Output SRT file path')
@click.option('-m', '--model', default='large-v3', help='Whisper model size')
@click.option('--device', default='cuda', help='Device (cuda/cpu)')
@click.option('--chunk-duration', default=30, help='Chunk duration in seconds')
def main(input_file, output, model, device, chunk_duration):
    """Transcribe audio/video files to SRT subtitles"""
    
    if output is None:
        output = Path(input_file).with_suffix('.srt')
    
    try:
        transcriber = Transcriber(model_size=model, device=device)
        result = transcriber.transcribe_file(
            input_file, 
            output_srt=output,
            chunk_duration=chunk_duration
        )
        
        print(f"\n✅ Transcription completed!")
        print(f"📄 Input: {input_file}")
        print(f"💾 Output: {result['srt_path']}")
        print(f"📊 Segments: {result['total_segments']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise click.Abort()

if __name__ == "__main__":
    main()