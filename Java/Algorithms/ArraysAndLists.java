package Java.Algorithms;

public class ArraysAndLists {
    public static void main(String args[]) {
        System.out.println("This java class will go over common array and list algorithms");

        oppositeEndTwoPointer();
    }

    /*
     * Two Pointer Implementation where two pointers are initialized at opposite ends of an array
     * O(n)
     * Example Provided: Valid, Case Insensitive Palindrome Ignoring Non-Alphanumeric Characters
     */
    private static void oppositeEndTwoPointer() {
        String palindrome = "A man, a plan, a canal: Panama";
        int left = 0;
        int right = palindrome.length() - 1;

        while(left < right) {
            while((left < right) && (!Character.isLetterOrDigit(palindrome.charAt(left)))) {
                left = left + 1;
            }

            while((left < right) && (Character.isLetterOrDigit(palindrome.charAt(right)))) {
                right = right - 1;
            }

            if (Character.toLowerCase(palindrome.charAt(left)) != Character.toLowerCase(palindrome.charAt(right))) {
                System.out.println("Not a valid palindrome");
                return;
            }

            left = left + 1;
            right = right - 1;
        }

        System.out.println("Valid Palindrome");
        return;
    }
}
