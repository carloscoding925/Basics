package Java.Algorithms;

import java.util.Arrays;

public class Sorting {
    public static void main(String args[]) {
        System.out.println("--- This java class will go over common sorting algorithms ---");

        int[] randArray = {64, 34, 25, 12, 22, 11, 90, 5, 123, 13, 97, 34, 98, 1, 99, 55, 25, 19, 67, 69, 100};

        bubbleSort(randArray);
        selectionSort(randArray);
        insertionSort(randArray);
        
        mergeSort(randArray, 0, randArray.length - 1);
        System.out.println("Sorting with Merge Sort");

        quickSort(randArray, 0, randArray.length - 1);
        System.out.println("Sorting with Quick Sort");

        heapSort(randArray);

        int[] countingArray = {4, 2, 2, 8, 3, 3, 1};
        countingSort(countingArray);

        return;
    }

    /*
     * Bubble Sort works by iterating through the entire array and comparing adjacent elements. Each pass
     * of Bubble Sort moves the largest item to its correct spot at the end of the sorted array, with items
     * being sorted from right to left (largest to smallest). Since each pass moves the largest usorted item
     * to its correct position, in our second for loop we can use length - i - 1 to shorten the amount of items
     * each iteration needs to check. If the array is sorted before the second for loop can run for the amount
     * of items in the array, the 'sorted' flag will be marked as 'false' and exit the algorithm early for a
     * faster run time.
     * 
     * Time Complexity of O(n^2) since two for loops are used for analyzing each item in the array
     * Space Complexity of O(1) since constant space is used (temp variables, flags, and fixed array)
     */
    private static void bubbleSort(int[] array) {
        int length = array.length;
        long startTime = System.nanoTime();

        for (int i = 0; i < length - 1; i++) {
            boolean swapped = false;
            for (int j = 0; j < length - i - 1; j++) {
                if (array[j] > array[j + 1]) {
                    int temp = array[j];
                    array[j] = array[j + 1];
                    array[j + 1] = temp;
                    swapped = true;
                }
            }
            if (!swapped) { break; }
        }

        long endTime = System.nanoTime();
        System.out.println("Sorting with Bubble Sort - Time to Execute: " + (endTime - startTime));

        return;
    }

    private static void selectionSort(int[] array) {
        int length = array.length;
        long startTime = System.nanoTime();

        for (int i = 0; i < length - 1; i++) {
            int min = i;
            for (int j = i + 1; j < length; j++) {
                if (array[j] < array[min]) {
                    min = j;
                }
            }

            int temp = array[min];
            array[min] = array[i];
            array[i] = temp;
        }

        long endTime = System.nanoTime();
        System.out.println("Sorting with Selection Sort - Time to Execute: " + (endTime - startTime));

        return;
    }

    private static void insertionSort(int[] array) {
        long startTime = System.nanoTime();
        for (int i = 1; i < array.length; i++) {
            int key = array[i];
            int j = i - 1;

            while(j >= 0 && array[j] > key) {
                array[j + 1] = array[j];
                j--;
            }
            array [j + 1] = key;
        }

        long endTime = System.nanoTime();
        System.out.println("Sorting with Insertion Sort - Time to Execute: " + (endTime - startTime));

        return;
    }

    private static void mergeSort(int[] array, int left, int right) {
        if (left < right) {
            int mid = left + (right - left) / 2;

            mergeSort(array, left, mid);
            mergeSort(array, mid + 1, right);

            merge(array, left, mid, right);
        }
    }

    private static void merge(int[] array, int left, int mid, int right) {
        int leftSize = mid - left + 1;
        int rightSize = right - mid;

        int[] leftArray = new int[leftSize];
        int[] rightArray = new int[rightSize];

        for (int i = 0; i < leftSize; i++) {
            leftArray[i] = array[left + 1];
        }

        for (int j = 0; j < rightSize; j++) {
            rightArray[j] = array[mid + 1 + j];
        }

        int i = 0, j = 0, k = left;

        while (i < leftSize && j < rightSize) {
            if (leftArray[i] <= rightArray[j]) {
                array[k] = leftArray[i];
                i++;
            }
            else {
                array[k] = rightArray[j];
                j++;
            }

            k++;
        }

        while (i < leftSize) {
            array[k] = leftArray[i];
            i++;
            k++;
        }

        while (j < rightSize) {
            array[k] = rightArray[j];
            j++;
            k++;
        }
    }

    private static void quickSort(int[] array, int left, int right) {
        if (left < right) {
            int pivotIndex = partition(array, left, right);

            quickSort(array, left, pivotIndex - 1);
            quickSort(array, pivotIndex + 1, right);
        }
    }

    private static int partition(int[] array, int left, int right) {
        int pivot = array[right];
        int i = left - 1;

        for (int j = left; j < right; j++) {
            if (array[j] < pivot) {
                i++;

                int temp = array[i];
                array[i] = array[j];
                array[j] = temp;
            }
        }

        int temp = array[i + 1];
        array[i + 1] = array[right];
        array[right] = temp;

        return i + 1;
    }

    private static void heapSort(int[] array) {
        System.out.println("Sorting with Heap Sort");
        int n = array.length;

        for (int i = n / 2 - 1; i >= 0; i--) {
            heapify(array, n, i);
        }

        for (int i = n - 1; i > 0; i--) {
            int temp = array[0];
            array[0] = array[i];
            array[i] = temp;

            heapify(array, i, 0);
        }

        return;
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

    private static int[] countingSort(int[] array) {
        System.out.println("Sorting with Counting Sort");

        int max = Arrays.stream(array).max().orElse(0);
        int min = Arrays.stream(array).min().orElse(0);
        int range = max - min + 1;

        int[] count = new int[range];
        int[] output = new int[array.length];

        for (int value : array) {
            count[value - min]++;
        }

        for (int i = 1; i < range; i++) {
            count[i] += count[i - 1];
        }

        for (int i = array.length - 1; i >= 0; i--) {
            output[count[array[i] - min] - 1] = array[i];
            count[array[i] - min]--;
        }

        return output;
    }
}
