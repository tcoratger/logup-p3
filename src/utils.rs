pub(crate) fn columns_to_row_major_flat<T: Clone>(columns: &[Vec<T>]) -> Vec<T> {
    let num_cols = columns.len();
    if num_cols == 0 {
        return Vec::new();
    }
    let num_rows = columns[0].len();
    let mut flat = Vec::with_capacity(num_rows * num_cols);

    for row in 0..num_rows {
        for col in columns {
            // Safety: row < columns[col].len() since all columns have num_rows elements
            flat.push(unsafe { col.get_unchecked(row).clone() });
        }
    }
    flat
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_single_column_single_row() {
        // Matrix view:
        //
        // [ 1 ]
        //
        // Only one column and one row. Flat vector should be [1].
        let columns = vec![vec![1]];
        let flat = columns_to_row_major_flat(&columns);
        assert_eq!(flat, vec![1]);
    }

    #[test]
    fn test_single_column_multiple_rows() {
        // Matrix view:
        //
        // [ 1 ]
        // [ 2 ]
        // [ 3 ]
        //
        // Only one column, so flat vector is simply [1, 2, 3].
        let columns = vec![vec![1, 2, 3]];
        let flat = columns_to_row_major_flat(&columns);
        assert_eq!(flat, vec![1, 2, 3]);
    }

    #[test]
    fn test_multiple_columns_single_row() {
        // Matrix view:
        //
        // [ 1  2  3 ]
        //
        // One row with three columns. Row-major flat vector: [1, 2, 3].
        let columns = vec![vec![1], vec![2], vec![3]];
        let flat = columns_to_row_major_flat(&columns);
        assert_eq!(flat, vec![1, 2, 3]);
    }

    #[test]
    fn test_multiple_columns_multiple_rows() {
        // Matrix view:
        //
        // [ 1  2  3 ]
        // [ 4  5  6 ]
        // [ 7  8  9 ]
        //
        // Columns (column-major input):
        // Col 0: [1, 4, 7]
        // Col 1: [2, 5, 8]
        // Col 2: [3, 6, 9]
        //
        // Row-major flat vector: [1, 2, 3, 4, 5, 6, 7, 8, 9].
        let columns = vec![vec![1, 4, 7], vec![2, 5, 8], vec![3, 6, 9]];
        let flat = columns_to_row_major_flat(&columns);
        assert_eq!(flat, vec![1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }

    #[test]
    fn test_columns_with_empty_inner_vectors() {
        // Matrix view:
        //
        // (empty)
        //
        // Columns exist but inner vectors are empty => zero rows.
        // Flat vector should be empty.
        let columns: Vec<Vec<i32>> = vec![vec![], vec![], vec![]];
        let flat = columns_to_row_major_flat(&columns);
        assert!(flat.is_empty());
    }
}
