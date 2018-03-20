import {NDArray, test_util, util} from 'deeplearn';

/**
 * Wraps dl.test_util.expectArraysClose to return only a subset of the original
 * error message. Helpful for very large arrays.
 * @param actual
 * @param expected
 * @param epsilon
 * @param numChars number of chars to return from the original error message.
 */
export function expectArraysClose(
    actual: NDArray|util.TypedArray, expected: NDArray|util.TypedArray,
    epsilon?: number, numChars = 1000) {
  try {
    test_util.expectArraysClose(actual, expected, epsilon);
  } catch (err) {
    const msg =
        err.message.length < 1000 ? err.message : err.message.substr(0, 1000);
    throw new Error(msg);
  }
}
