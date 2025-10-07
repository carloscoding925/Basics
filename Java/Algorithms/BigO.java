package Java.Algorithms;

import java.util.Arrays;

public class BigO {
    /*
     * For inputs 10^5, need O(n log n) or better
     * For inputs 10^4, need O(n^2) or better
     */
    public static void main(String[] args) {
        System.out.println("--- This java class will go over common Big O algorithms ---");
        int[] sampleArray = {0, 1, 2, 3, 4, 5};
        int[] smallArray = {0, 1, 2};

        // O(1);
        constantTime();
        // O(logn)
        logarithmicTime();
        // O(n)
        linearTime();
        // O(nlogn)
        nlognTime(sampleArray);
        // O(n^2)
        nSquaredTime();
        // O(2^n)
        fibonacciExponential(5);
        // O(n!)
        factorialGeneratePermutations(smallArray, 0);

        return;
    }

    private static void constantTime() {
        int[] array = {0, 1, 2, 3};

        if (array == null || array.length == 0) {
            throw new IllegalArgumentException("Array is Empty");
        }

        System.out.println("First Element: " + array[0]);

        return;
    }

    private static void logarithmicTime() {
        int[] array = {-3, -2, -1, 0, 1, 2, 3, 4};
        int left = 0;
        int right = array.length - 1;
        int target = 2;

        while (left <= right) {
            int mid = left + (right - left) / 2;

            if (array[mid] == target) {
                System.out.println("Found Target at Index: " + mid);
                return;
            }
            else if (array[mid] < target) {
                left = mid + 1;
            }
            else {
                right = mid - 1;
            }
        }

        System.out.println("Did not find target");
        return;
    }

    private static void linearTime() {
        int[] array = {0, 1, 2, 3, 4, 5};
        int max = array[0];

        for (int i = 0; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
            }
        }

        System.out.println("Largest item in index: " + max);
        return;
    }

    private static void nlognTime(int[] array) {
        System.out.println("Heap Sort for O(nlogn) time");
        int length = array.length;

        for (int i = length / 2 - 1; i >= 0; i--) {
            heapify(array, length, i);
        }

        for (int i = length - 1; i > 0; i--) {
            int temp = array[0];
            array[0] = array[i];
            array[i] = temp;

            heapify(array, i, 0);
        }
    }

    private static void heapify(int[] array, int n, int i) {
        int largest = i;
        int left = 2 * i + 1;
        int right = 2 * i + 2;

        if (left < n && array[left] > array[largest]) {
            largest = left;
        }

        if (right < n && array[right] > array[largest]) {
            largest = right;
        }

        if (largest != i) {
            int swap = array[i];
            array[i] = array[largest];
            array[largest] = swap;

            heapify(array, n, largest);
        }
    }

    private static void nSquaredTime() {
        int[] array = {0, 3, 6, 1, 2, 9, 3, 5};

        for (int i = 0; i < array.length - 1; i++) {
            for (int j = 0; j < array.length - i - 1; j++) {
                if (array[j] > array[j + 1]) {
                    int temp = array[j];
                    array[j] = array[j + 1];
                    array[j + 1] = temp;
                }
            }
        }

        System.out.println("Sorted Array using Bubble Sort");
        return;
    }

    private static int fibonacciExponential(int n) {
        System.out.println("Computing Fibonacci for: " + n);
        if (n <= 1) {
            return n;
        }

        return fibonacciExponential(n - 1) + fibonacciExponential(n - 2);
    }

    private static void factorialGeneratePermutations(int[] array, int start) {
        if (start == array.length - 1) {
            System.out.println(Arrays.toString(array));
            return;
        }

        for (int i = start; i < array.length; i++) {
            swap(array, start, i);
            factorialGeneratePermutations(array, start + 1);
            swap(array, start, i);
        }
    }

    private static void swap(int[] array, int i, int j) {
        int temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}
