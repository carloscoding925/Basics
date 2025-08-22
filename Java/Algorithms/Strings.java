package Java.Algorithms;

public class Strings {
    public static void main(String args[]) {
        System.out.println("--- This java class will go over common string manipulation algorithms ---");

        stringConcatenation();
        characterArray();
        stringBuilderMethod();

        return;
    }

    /*
     * String Concatenation where a new string is created each time a character is appended
     * instead of modifying the existing string because strings are immutable 
     * Bad Practice - Results in Time Complexity of O(n^2)
     */
    private static void stringConcatenation() {
        String input = "original";
        String result = "";

        long startTime = System.nanoTime();
        for (char character : input.toCharArray()) {
            result = result + character;
        }
        long endTime = System.nanoTime();

        System.out.println("String Concatenation Result: " + result + " Time to Execute: " + (endTime - startTime));
        return;
    }

    /*
     * Building a new character array of the size of the original string, filling it with an
     * equivalent amount of characters and converting it into a String is much faster
     * Better Practice - O(n)
     */
    private static void characterArray() {
        String input = "original";
        char[] result = new char[input.length()];

        long startTime = System.nanoTime();
        for (int i = 0; i < input.length(); i++) {
            result[i] = input.charAt(i);
        }
        String resultString = new String(result);
        long endTime = System.nanoTime();

        System.out.println("Character Array Result: " + resultString + " Time to Execute: " + (endTime - startTime));
        return;
    }

    /*
     * This is the standard Java approach to creating new strings with similar time complexity
     * to that of the character array.
     * Best Practice - O(n)
     */
    private static void stringBuilderMethod() {
        String input = "original";
        StringBuilder builder = new StringBuilder();

        long startTime = System.nanoTime();
        for (char character : input.toCharArray()) {
            builder.append(character);
        }
        String builderResult = builder.toString();
        long endTime = System.nanoTime();

        System.out.println("String Builder Result: " + builderResult + " Time to Execute: " + (endTime - startTime));
        return;
    }
}
