package Java.Algorithms;

public class Sorting {
    public static void main(String args[]) {
        System.out.println("--- This java class will go over common sorting algorithms ---");

        int[] randArray = {64, 34, 25, 12, 22, 11, 90, 5, 123, 13, 97, 34, 98, 1, 99, 55, 25, 19, 67, 69, 100};

        bubbleSort(randArray);

        return;
    }

    /*
     * Bubble Sort
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
}
