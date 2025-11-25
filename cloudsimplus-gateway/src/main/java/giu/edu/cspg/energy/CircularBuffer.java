package giu.edu.cspg.energy;

import java.util.ArrayList;
import java.util.List;

/**
 * Generic circular buffer for storing historical data with fixed capacity.
 * When buffer is full, oldest elements are overwritten.
 *
 * @param <T> Type of elements to store
 */
public class CircularBuffer<T> {
    private final Object[] buffer;
    private int head = 0;  // Index of oldest element
    private int size = 0;  // Current number of elements
    private final int capacity;

    /**
     * Create a circular buffer with specified capacity.
     *
     * @param capacity Maximum number of elements to store
     */
    public CircularBuffer(int capacity) {
        if (capacity <= 0) {
            throw new IllegalArgumentException("Capacity must be positive");
        }
        this.capacity = capacity;
        this.buffer = new Object[capacity];
    }

    /**
     * Add an element to the buffer.
     * If buffer is full, overwrites the oldest element.
     *
     * @param element Element to add
     */
    public void add(T element) {
        if (element == null) {
            throw new IllegalArgumentException("Cannot add null element");
        }

        int index = (head + size) % capacity;
        buffer[index] = element;

        if (size < capacity) {
            size++;
        } else {
            head = (head + 1) % capacity;
        }
    }

    /**
     * Get the last N elements from the buffer (most recent).
     *
     * @param n Number of elements to retrieve
     * @return List of last n elements, or all elements if n > size
     */
    @SuppressWarnings("unchecked")
    public List<T> getLast(int n) {
        int actualN = Math.min(n, size);
        List<T> result = new ArrayList<>(actualN);

        for (int i = size - actualN; i < size; i++) {
            int index = (head + i) % capacity;
            result.add((T) buffer[index]);
        }

        return result;
    }

    /**
     * Get element at relative index (0 = oldest, size-1 = newest).
     *
     * @param index Relative index
     * @return Element at index
     */
    @SuppressWarnings("unchecked")
    public T get(int index) {
        if (index < 0 || index >= size) {
            throw new IndexOutOfBoundsException("Index: " + index + ", Size: " + size);
        }

        int actualIndex = (head + index) % capacity;
        return (T) buffer[actualIndex];
    }

    /**
     * Get current number of elements in buffer.
     *
     * @return Number of elements
     */
    public int size() {
        return size;
    }

    /**
     * Get buffer capacity.
     *
     * @return Maximum capacity
     */
    public int capacity() {
        return capacity;
    }

    /**
     * Check if buffer is empty.
     *
     * @return true if empty
     */
    public boolean isEmpty() {
        return size == 0;
    }

    /**
     * Check if buffer is full.
     *
     * @return true if full
     */
    public boolean isFull() {
        return size == capacity;
    }

    /**
     * Clear all elements from buffer.
     */
    public void clear() {
        for (int i = 0; i < capacity; i++) {
            buffer[i] = null;
        }
        head = 0;
        size = 0;
    }

    /**
     * Get all elements as a list (oldest to newest).
     *
     * @return List of all elements
     */
    @SuppressWarnings("unchecked")
    public List<T> toList() {
        List<T> result = new ArrayList<>(size);
        for (int i = 0; i < size; i++) {
            int index = (head + i) % capacity;
            result.add((T) buffer[index]);
        }
        return result;
    }

    @Override
    public String toString() {
        return String.format("CircularBuffer[size=%d, capacity=%d, head=%d]",
                           size, capacity, head);
    }
}
