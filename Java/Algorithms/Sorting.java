package Java.Algorithms;

public class Sorting {
    public static void main(String args[]) {
        System.out.println("--- This java class will go over common sorting algorithms ---");

        int[] randArray = {64, 34, 25, 12, 22, 11, 90, 5, 123, 13, 97, 34, 98, 1, 99, 55, 25, 19, 67, 69, 100};

        bubbleSort(randArray);
        selectionSort(randArray);

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

        }

        long endTime = System.nanoTime();
        System.out.println("Sorting with Insertion Sort - Time to Execute: " + (endTime - startTime));

        return;
    }
}
