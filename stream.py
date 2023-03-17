from absl import app
from absl import flags
from absl import logging
from functools import partial
from functools import wraps
import numpy as np
import queue
import sounddevice as sd
from time import time as now
import whisper

FLAGS = flags.FLAGS
flags.DEFINE_string('model_name', 'base.en',
                    'The version of the OpenAI Whisper model to use.')
flags.DEFINE_string('language', 'en',
                    'The language to use or None to auto-detect.')
flags.DEFINE_string('input_device', 'plughw:2,0',
                    'The input device used to record audio.')
flags.DEFINE_integer('sample_rate', 16000,
                     'The sample rate of the recorded audio.')
flags.DEFINE_integer('num_channels', 1,
                     'The number of channels of the recorded audio.')
flags.DEFINE_integer('channel_index', 0,
                     'The index of the channel to use for transcription.')
flags.DEFINE_integer('chunk_seconds', 10,
                     'The length in seconds of each recorded chunk of audio.')
flags.DEFINE_string('latency', 'low', 'The latency of the recording stream.')


# A decorator to log the timing of performance-critical functions.
def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = now()
        result = func(*args, **kwargs)
        stop = now()
        logging.debug(f'{func.__name__} took {stop-start:.3f}s')
        return result
    return wrapper


@timed
def transcribe(model, audio):
    # Run the Whisper model to transcribe the audio chunk.
    result = whisper.transcribe(model=model, audio=audio)

    # Use the transcribed text.
    text = result['text'].strip()
    logging.info(text)


@timed
def stream_callback(indata, frames, time, status, audio_queue):
    if status:
        logging.warning(f'Stream callback status: {status}')

    # Add this chunk of audio to the queue.
    audio = indata[:, FLAGS.channel_index].copy()
    audio_queue.put(audio)


@timed
def process_audio(audio_queue, model):
    # Block until the next chunk of audio is available on the queue.
    audio = audio_queue.get()

    # Transcribe the latest audio chunk.
    transcribe(model=model, audio=audio)


def main(argv):
    # Load the Whisper model into memory, downloading first if necessary.
    logging.info(f'Loading model "{FLAGS.model_name}"...')
    model = whisper.load_model(name=FLAGS.model_name)

    # The first run of the model is slow (buffer init), so run it once empty.
    logging.info('Warming model up...')
    block_size = FLAGS.chunk_seconds * FLAGS.sample_rate
    whisper.transcribe(model=model,
                       audio=np.zeros(block_size, dtype=np.float32))

    # Stream audio chunks into a queue and process them from there. The
    # callback is running on a separate thread.
    logging.info('Starting stream...')
    audio_queue = queue.Queue()
    callback = partial(stream_callback, audio_queue=audio_queue)
    with sd.InputStream(samplerate=FLAGS.sample_rate,
                        blocksize=block_size,
                        device=FLAGS.input_device,
                        channels=FLAGS.num_channels,
                        dtype=np.float32,
                        latency=FLAGS.latency,
                        callback=callback):
        while True:
            # Process chunks of audio from the queue.
            process_audio(audio_queue, model)


if __name__ == '__main__':
    app.run(main)
