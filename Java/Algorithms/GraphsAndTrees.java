package Java.Algorithms;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;
import java.util.Set;

public class GraphsAndTrees {
    public static void main(String[] args) {
        System.out.println("--- This java class will go over common graph and tree algorithms ---");

        treeBfs();

        TreeNode root = new TreeNode(1);
        root.leftNode = new TreeNode(2);
        root.rightNode = new TreeNode(3);
        root.leftNode.leftNode = new TreeNode(4);
        root.leftNode.rightNode = new TreeNode(5);
        int target = 4;
        treeDfs(root, target);
        
        graphBfs();

        GraphNode node1 = new GraphNode(1);
        GraphNode node2 = new GraphNode(2);
        GraphNode node3 = new GraphNode(3);
        GraphNode node4 = new GraphNode(4);
        node1.neighbors.add(node2);
        node1.neighbors.add(node3);
        node2.neighbors.add(node1);
        node2.neighbors.add(node4);
        node3.neighbors.add(node1);
        node4.neighbors.add(node2);
        Set<GraphNode> visited = new HashSet<>();
        graphDfs(node1, visited);

        return;
    }

    static class TreeNode {
        int val;
        TreeNode leftNode;
        TreeNode rightNode;

        TreeNode(int val) {
            this.val = val;
        }

        TreeNode(int val, TreeNode leftNode, TreeNode rightNode) {
            this.val = val;
            this.leftNode = leftNode;
            this.rightNode = rightNode;
        }
    }

    static class GraphNode {
        int val;
        List<GraphNode> neighbors;

        GraphNode(int val) {
            this.val = val;
            this.neighbors = new ArrayList<>();
        }
    }

    /*
     * Breadth First Search for Trees
     * O(n) Time Complexity, O(w) Space Complexity where w is the maximum width
     * Example Provided: Find a target node value using BFS
     */
    private static void treeBfs() {
        TreeNode root = new TreeNode(1);
        root.leftNode = new TreeNode(2);
        root.rightNode = new TreeNode(3);
        root.leftNode.leftNode = new TreeNode(4);
        root.leftNode.rightNode = new TreeNode(5);

        Queue<TreeNode> queue = new LinkedList<>();
        int target = 4;

        queue.offer(root);

        while (!queue.isEmpty()) {
            TreeNode node = queue.poll();
            System.out.println("Visiting Tree Node: " + node.val);

            if (node.val == target) {
                System.out.println("Found Node with Target: " + node.val);
                return;
            }

            if (node.leftNode != null) {
                queue.offer(node.leftNode);
            }

            if (node.rightNode != null) {
                queue.offer(node.rightNode);
            }
        }

        System.out.println("Target BFS Node Not Found");
        return;
    }

    /*
     * Depth First Search for Trees
     * O(n) Time Complexity, O(h) Space Complexity where h is the height of the tree
     * Example Provided: Find a target node value using DFS using Recursion
     */
    private static TreeNode treeDfs(TreeNode root, int target) {
        if (root == null) {
            System.out.println("Tree is Empty");
            return null;
        }

        System.out.println("Visiting Tree Node: " + root.val);
        if (root.val == target) {
            System.out.println("Found Node with Target: " + root.val);
            return root;
        }

        TreeNode leftResult = treeDfs(root.leftNode, target);
        if (leftResult != null) {
            return leftResult;
        }

        TreeNode rightResult = treeDfs(root.rightNode, target);
        if (rightResult != null) {
            return rightResult;
        }

        System.out.println("Node with value: " + target + " not found.");
        return null;
    }

    /*
     * Breadth First Search on Graphs
     * O(V + E) Time Complexity, O(V) Space Complexity where V is the vertices and E is the edges
     * Use Case: Grids, Adjacency Lists, Networks, Where Structure contains cycles/duplicate paths or
     * you need to find the shortest number of steps
     * Example Provided: Visit all nodes via BFS
     */
    private static void graphBfs() {
        GraphNode nodeOne = new GraphNode(1);
        GraphNode nodeTwo = new GraphNode(2);
        GraphNode nodeThree = new GraphNode(3);
        GraphNode nodeFour = new GraphNode(4);

        nodeOne.neighbors.add(nodeTwo);
        nodeOne.neighbors.add(nodeThree);
        nodeTwo.neighbors.add(nodeOne);
        nodeTwo.neighbors.add(nodeFour);
        nodeThree.neighbors.add(nodeOne);
        nodeFour.neighbors.add(nodeTwo);

        Queue<GraphNode> graphQueue = new LinkedList<>();
        Set<GraphNode> visited = new HashSet<>();

        graphQueue.offer(nodeOne);
        visited.add(nodeOne);

        while (!graphQueue.isEmpty()) {
            GraphNode node = graphQueue.poll();
            System.out.println("Visiting Node with Value: " + node.val);

            for (GraphNode neighbor : node.neighbors) {
                if (visited.contains(neighbor)) {
                    continue;
                }
                graphQueue.offer(neighbor);
                visited.add(neighbor);
            }
        }

        System.out.println("Visited all Graph Nodes via BFS");
        return;
    }

    /*
     * Depth First Search on Graphs
     * O(V + E) Time Complexity, O(V) Space Complexity where V are the vertices and E are the edges
     * Use Case: Exploring all Paths, Detecting Cycles, Topological Sorting, Maze Solving, Component Connecting,
     * When you want to go as deep as possible before backtracking
     * Example Provided: Visit all Nodes via DFS
     */
    private static void graphDfs(GraphNode root, Set<GraphNode> visited) {
        if (root == null) {
            System.out.println("Root is empty");
            return;
        }
        
        System.out.println("Visiting Node with Value: " + root.val);

        for (GraphNode neighbor : root.neighbors) {
            if (visited.contains(neighbor)) {
                continue;
            }

            visited.add(neighbor);
            graphDfs(neighbor, visited);
        }
    }
}
