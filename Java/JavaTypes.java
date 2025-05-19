package Java;

public class JavaTypes {
    public static void main(String[] args) {
        System.out.println("Examining Java Types");
    }

    private static void dataTypes() {
        // Primitive data types
        int integer = 1; // Integer variable
        double decimal = 1.1; // Decimal variable
        char character = 'a'; // Character variable
        String string = "Hello"; // String variable
        boolean bool = true; // Boolean variable

        // Not so common
        byte byteVar = 1; // Byte variable
        short shortVar = 1; // Short variable
        long longVar = 1; // Long variable
        float floatVar = 1.1f; // Float variable

        final int constantInt = 1; // Final (Constant) variables cannot be changed

        // Array declarations
        int[] array = {1, 2, 3}; // Array variable
        char[] charArray = {'a', 'b', 'c'}; // Character array variable
        String[] stringArray = {"Hello", "World"}; // String array variable
        String[][] string2DArray = {{"Hello", "World"}, {"Java", "Programming"}}; // 2D String array variable

        String[] stringArray2 = new String[5]; // Array of Strings with 5 elements

        double sampleDouble = 5.2;
        int sampleInt = (int) sampleDouble; // Casting a double to an int
    }
}
