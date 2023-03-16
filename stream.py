from absl import app
from absl import flags
from absl import logging
import numpy as np
import sounddevice as sd
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
flags.DEFINE_integer('chunk_seconds', 30,
                     'The length in seconds of each recorded chunk of audio.')


def main(argv):
    # Download the Whisper model into memory.
    logging.info(f'Loading model "{FLAGS.model_name}" ...')
    model = whisper.load_model(name=FLAGS.model_name)

    # Allocate a buffer for chunks of audio to be recorded into.
    audio = np.zeros(
        [FLAGS.chunk_seconds * FLAGS.sample_rate, FLAGS.num_channels],
        dtype=np.float32)

    # Run the model once on empty audio to warm it up.
    logging.info('Warming up the model ...')
    whisper.transcribe(model=model, audio=audio.squeeze())

    # TODO: Use sounddevice.InputStream
    # TODO: Use overlapping chunks

    # Record a chunk of audio.
    logging.info('Recording audio ...')
    sd.rec(device=FLAGS.input_device, samplerate=FLAGS.sample_rate,
           out=audio, blocking=True)

    # Transcribe the recorded audio.
    logging.info('Transcribing audio ...')
    result = whisper.transcribe(model=model, audio=audio.squeeze())
    text = result['text'].strip()

    # Display the transcribed text.
    logging.info(text)


if __name__ == '__main__':
    app.run(main)
