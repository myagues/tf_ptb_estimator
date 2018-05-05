"""Tests for models.tutorials.rnn.ptb.reader."""

import os.path

import tensorflow as tf

import reader


class PtbReaderTest(tf.test.TestCase):

    def setUp(self):
        self._string_data = "\n".join(
            [" hello there i am",
             " rain as day",
             " want some cheesy puffs ?"])

    def testPtbRawData(self):
        tmpdir = tf.test.get_temp_dir()
        for suffix in "train", "valid", "test":
            filename = os.path.join(tmpdir, "ptb.%s.txt" % suffix)
            with tf.gfile.GFile(filename, "w") as fh:
                fh.write(self._string_data)
        # Smoke test
        output = reader.ptb_raw_data(tmpdir)
        self.assertEqual(len(output), 4)

    def testPtbProducer(self):
        raw_data = [4, 3, 2, 1, 0, 5, 6, 1, 1, 1, 1, 0, 3, 4, 1]
        batch_size = 3
        num_steps = 2
        dataset = reader.ptb_producer(raw_data, batch_size, num_steps)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        with self.test_session() as session:
            xval, yval = session.run(next_element)
            self.assertAllEqual(xval, [[4, 3], [5, 6], [1, 0]])
            self.assertAllEqual(yval, [[3, 2], [6, 1], [0, 3]])
            xval, yval = session.run(next_element)
            self.assertAllEqual(xval, [[2, 1], [1, 1], [3, 4]])
            self.assertAllEqual(yval, [[1, 0], [1, 1], [4, 1]])


if __name__ == "__main__":
    tf.test.main()
