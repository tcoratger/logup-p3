use itertools::Itertools;
use p3_field::{ExtensionField, Field, PrimeField};
use p3_matrix::{Matrix, dense::RowMajorMatrix};

/// A two-dimensional representation of an execution trace of a STARK protocol.
///
/// This trace consists of a main table (base field) and an auxiliary table (extension field),
/// typically used for enforcing permutation arguments or other auxiliary constraints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TraceTable<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    /// Main execution trace (base field columns).
    pub main_table: RowMajorMatrix<F>,

    /// Auxiliary columns used for permutation or lookup constraints (extension field).
    pub aux_table: RowMajorMatrix<EF>,
}

impl<F, EF> TraceTable<F, EF>
where
    F: Field + PrimeField,
    EF: ExtensionField<F>,
{
    /// Creates a new `TraceTable` from column vectors.
    ///
    /// # Arguments
    /// - `main_columns`: Columns over the base field F.
    /// - `aux_columns`: Columns over the extension field EF.
    /// - `step_size`: Number of rows corresponding to one logical program step.
    ///
    /// Columns are flattened internally into row-major form for efficient indexing.
    #[must_use]
    pub fn from_columns(main_columns: &[Vec<F>], aux_columns: &[Vec<EF>]) -> Self {
        let num_rows = main_columns.first().map_or(0, Vec::len);
        let num_main_cols = main_columns.len();
        let num_aux_cols = aux_columns.len();

        let flat_main = (0..num_rows)
            .flat_map(|i| main_columns.iter().map(move |col| col[i]))
            .collect();

        let flat_aux = (0..num_rows)
            .flat_map(|i| aux_columns.iter().map(move |col| col[i]))
            .collect();

        Self {
            main_table: RowMajorMatrix::new(flat_main, num_main_cols),
            aux_table: RowMajorMatrix::new(flat_aux, num_aux_cols),
        }
    }

    /// Constructs a LogUp read-only trace table from addresses and values.
    ///
    /// This method:
    /// - Sorts `(address, value)` pairs,
    /// - Computes multiplicities,
    /// - Pads sorted columns to match original length,
    /// - Initializes auxiliary columns to zero.
    #[must_use]
    pub fn read_only_logup_trace(addresses: &[F], values: &[F]) -> Self {
        let mut pairs: Vec<_> = addresses.iter().zip(values).collect();
        pairs.sort_by_key(|&(a, _)| *a);

        let mut sorted_addrs = Vec::new();
        let mut sorted_vals = Vec::new();
        let mut multiplicities = Vec::new();

        for (key, group) in &pairs.into_iter().chunk_by(|&(a, v)| (a, v)) {
            let group_count = group.count();
            sorted_addrs.push(*key.0);
            sorted_vals.push(*key.1);
            multiplicities.push(F::from_usize(group_count));
        }

        let pad_len = addresses.len();
        let pad_value_addr = *sorted_addrs.last().unwrap();
        let pad_value_val = *sorted_vals.last().unwrap();

        sorted_addrs.resize(pad_len, pad_value_addr);
        sorted_vals.resize(pad_len, pad_value_val);
        multiplicities.resize(pad_len, F::ZERO);

        let main_columns = [
            addresses.to_owned(),
            values.to_owned(),
            sorted_addrs,
            sorted_vals,
            multiplicities,
        ];

        let zero_aux = EF::zero_vec(pad_len);

        Self::from_columns(&main_columns, &[zero_aux])
    }

    /// Builds the auxiliary "s" column for permutation arguments (LogUp).
    ///
    /// Update rule:
    ///
    /// ```text
    /// s₀ = m₀ / (z - (a'₀ + α v'₀)) - 1 / (z - (a₀ + α v₀))
    /// sᵢ = sᵢ₋₁ + mᵢ / (z - (a'ᵢ + α v'ᵢ)) - 1 / (z - (aᵢ + α vᵢ))
    /// ```
    pub fn build_auxiliary_column(&mut self, z: EF, alpha: EF) {
        let n = self.main_table.height();
        let main = &self.main_table;

        let mut s = Vec::with_capacity(n);

        let extract = |i, j| unsafe { main.get_unchecked(i, j) };
        let term = |a: F, v: F| (z - (alpha * v + a)).inverse();

        let a0 = extract(0, 0);
        let v0 = extract(0, 1);
        let a_sorted0 = extract(0, 2);
        let v_sorted0 = extract(0, 3);
        let m0 = extract(0, 4);

        let unsorted_inv0 = term(a0, v0);
        let sorted_inv0 = term(a_sorted0, v_sorted0);
        s.push(sorted_inv0 * m0 - unsorted_inv0);

        for i in 1..n {
            let a = extract(i, 0);
            let v = extract(i, 1);
            let a_sorted = extract(i, 2);
            let v_sorted = extract(i, 3);
            let m = extract(i, 4);

            let unsorted_inv = term(a, v);
            let sorted_inv = term(a_sorted, v_sorted);

            let si = s[i - 1] + sorted_inv * m - unsorted_inv;
            s.push(si);
        }

        for (i, &val) in s.iter().enumerate() {
            self.aux_table.values[i] = val;
        }
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};
    use p3_matrix::Matrix;

    use super::*;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    fn get_column<T>(matrix: &RowMajorMatrix<T>, col_idx: usize) -> Vec<T>
    where
        T: Clone + Send + Sync,
    {
        let mut col = Vec::with_capacity(matrix.height());
        let width = matrix.width();
        for row in 0..matrix.height() {
            col.push(matrix.values[row * width + col_idx].clone());
        }
        col
    }

    #[test]
    fn test_read_only_logup_trace_single_entry() {
        let addresses = [F::from_u64(10)];
        let values = [F::from_u64(42)];

        let trace = TraceTable::<F, EF>::read_only_logup_trace(&addresses, &values);

        // Expected logical contents for each column (row 0):
        //
        // addresses      = 10
        // values         = 42
        // sorted_addrs   = 10
        // sorted_vals    = 42
        // multiplicities = 1

        let expected_main = vec![
            addresses[0], // addresses
            values[0],    // values
            addresses[0], // sorted addresses
            values[0],    // sorted values
            F::ONE,       // multiplicities
        ];

        assert_eq!(trace.main_table.height(), 1);
        assert_eq!(trace.main_table.width(), 5);
        assert_eq!(trace.main_table.values, expected_main);
        assert_eq!(trace.aux_table.values, vec![EF::ZERO]);
    }

    #[test]
    fn test_read_only_logup_trace_multiple_unique() {
        let addresses = [F::from_u64(30), F::from_u64(10), F::from_u64(20)];
        let values = [F::from_u64(3), F::from_u64(1), F::from_u64(2)];

        let trace = TraceTable::<F, EF>::read_only_logup_trace(&addresses, &values);

        // Sorted pairs: (10, 1), (20, 2), (30, 3)
        //
        // Final table logic:
        //
        // Row 0: [30, 3, 10, 1, 1] ← original (30,3), sorted (10,1)
        // Row 1: [10, 1, 20, 2, 1] ← original (10,1), sorted (20,2)
        // Row 2: [20, 2, 30, 3, 1] ← original (20,2), sorted (30,3)
        //
        // Columns are: [address, value, sorted_addr, sorted_val, multiplicity]

        let expected_main = vec![
            // Row 0
            F::from_u64(30),
            F::from_u64(3),
            F::from_u64(10),
            F::from_u64(1),
            F::ONE,
            // Row 1
            F::from_u64(10),
            F::from_u64(1),
            F::from_u64(20),
            F::from_u64(2),
            F::ONE,
            // Row 2
            F::from_u64(20),
            F::from_u64(2),
            F::from_u64(30),
            F::from_u64(3),
            F::ONE,
        ];

        assert_eq!(trace.main_table.height(), 3);
        assert_eq!(trace.main_table.width(), 5);
        assert_eq!(trace.main_table.values, expected_main);
        assert_eq!(trace.aux_table.values, vec![EF::ZERO, EF::ZERO, EF::ZERO]);
    }

    #[test]
    fn test_read_only_logup_trace_repeated_entries() {
        let addresses = [F::from_u64(10), F::from_u64(10), F::from_u64(20)];
        let values = [F::from_u64(5), F::from_u64(5), F::from_u64(7)];

        let trace = TraceTable::<F, EF>::read_only_logup_trace(&addresses, &values);

        // Sorted pairs: (10, 5) with multiplicity 2, (20, 7) with multiplicity 1
        //
        // Final table logic:
        //
        // Row 0: [10, 5, 10, 5, 2] ← original (10,5), sorted (10,5), multiplicity 2
        // Row 1: [10, 5, 20, 7, 1] ← original (10,5), sorted (20,7), multiplicity 1
        // Row 2: [20, 7, 20, 7, 0] ← original (20,7), sorted (20,7), multiplicity 0
        //
        // Columns are: [address, value, sorted_addr, sorted_val, multiplicity]

        let expected_main = vec![
            // Row 0
            F::from_u64(10),
            F::from_u64(5),
            F::from_u64(10),
            F::from_u64(5),
            F::from_u64(2),
            // Row 1
            F::from_u64(10),
            F::from_u64(5),
            F::from_u64(20),
            F::from_u64(7),
            F::from_u64(1),
            // Row 2
            F::from_u64(20),
            F::from_u64(7),
            F::from_u64(20),
            F::from_u64(7),
            F::ZERO,
        ];

        assert_eq!(trace.main_table.height(), 3);
        assert_eq!(trace.main_table.width(), 5);
        assert_eq!(trace.main_table.values, expected_main);
        assert_eq!(trace.aux_table.values, vec![EF::ZERO, EF::ZERO, EF::ZERO]);
    }

    #[test]
    fn test_logup_trace_all_unique() {
        let addresses = vec![
            F::from_u64(3),
            F::from_u64(7),
            F::from_u64(2),
            F::from_u64(8),
            F::from_u64(4),
            F::from_u64(5),
            F::from_u64(1),
            F::from_u64(6),
        ];
        let values = vec![
            F::from_u64(30),
            F::from_u64(70),
            F::from_u64(20),
            F::from_u64(80),
            F::from_u64(40),
            F::from_u64(50),
            F::from_u64(10),
            F::from_u64(60),
        ];

        let trace = TraceTable::<F, EF>::read_only_logup_trace(&addresses, &values);

        let expected_sorted_addresses = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
            F::from_u64(5),
            F::from_u64(6),
            F::from_u64(7),
            F::from_u64(8),
        ];
        let expected_sorted_values = vec![
            F::from_u64(10),
            F::from_u64(20),
            F::from_u64(30),
            F::from_u64(40),
            F::from_u64(50),
            F::from_u64(60),
            F::from_u64(70),
            F::from_u64(80),
        ];
        let expected_multiplicities = vec![F::ONE; 8];

        assert_eq!(get_column(&trace.main_table, 0), addresses);
        assert_eq!(get_column(&trace.main_table, 1), values);
        assert_eq!(get_column(&trace.main_table, 2), expected_sorted_addresses);
        assert_eq!(get_column(&trace.main_table, 3), expected_sorted_values);
        assert_eq!(get_column(&trace.main_table, 4), expected_multiplicities);
    }

    #[test]
    fn test_logup_trace_with_repeats() {
        let addresses = vec![
            F::from_u64(3), // a0
            F::from_u64(2), // a1
            F::from_u64(2), // a2
            F::from_u64(3), // a3
            F::from_u64(4), // a4
            F::from_u64(5), // a5
            F::from_u64(1), // a6
            F::from_u64(3), // a7
        ];
        let values = vec![
            F::from_u64(30), // v0
            F::from_u64(20), // v1
            F::from_u64(20), // v2
            F::from_u64(30), // v3
            F::from_u64(40), // v4
            F::from_u64(50), // v5
            F::from_u64(10), // v6
            F::from_u64(30), // v7
        ];

        let trace = TraceTable::<F, EF>::read_only_logup_trace(&addresses, &values);

        let expected_sorted_addresses = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(4),
            F::from_u64(5),
            F::from_u64(5),
            F::from_u64(5),
            F::from_u64(5),
        ];
        let expected_sorted_values = vec![
            F::from_u64(10),
            F::from_u64(20),
            F::from_u64(30),
            F::from_u64(40),
            F::from_u64(50),
            F::from_u64(50),
            F::from_u64(50),
            F::from_u64(50),
        ];
        let expected_multiplicities = vec![
            F::ONE,
            F::from_u64(2),
            F::from_u64(3),
            F::ONE,
            F::ONE,
            F::ZERO,
            F::ZERO,
            F::ZERO,
        ];

        assert_eq!(get_column(&trace.main_table, 0), addresses);
        assert_eq!(get_column(&trace.main_table, 1), values);
        assert_eq!(get_column(&trace.main_table, 2), expected_sorted_addresses);
        assert_eq!(get_column(&trace.main_table, 3), expected_sorted_values);
        assert_eq!(get_column(&trace.main_table, 4), expected_multiplicities);
    }

    #[test]
    fn test_build_auxiliary_column_example() {
        use p3_baby_bear::BabyBear;
        use p3_field::extension::BinomialExtensionField;

        type F = BabyBear;
        type EF = BinomialExtensionField<F, 4>;

        // ------------------------------------------------------------------------------------
        // Define example main trace
        // Columns: a, v, a', v', m
        // ------------------------------------------------------------------------------------
        let a = vec![
            F::from_u64(3),
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(2),
        ];
        let v = vec![
            F::from_u64(30),
            F::from_u64(10),
            F::from_u64(20),
            F::from_u64(20),
        ];
        let a_sorted = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(3),
            F::from_u64(3), // Padding row with zero multiplicity
        ];
        let v_sorted = vec![
            F::from_u64(10),
            F::from_u64(20),
            F::from_u64(30),
            F::from_u64(30), // Padding row with zero multiplicity
        ];
        let m = vec![
            F::from_u64(1),
            F::from_u64(2),
            F::from_u64(1),
            F::from_u64(0), // Padding row with zero multiplicity
        ];

        let mut trace = TraceTable::<F, EF>::read_only_logup_trace(&a, &v);

        assert_eq!(trace.aux_table.values, vec![EF::ZERO; 4]);
        assert_eq!(get_column(&trace.main_table, 0), a);
        assert_eq!(get_column(&trace.main_table, 1), v);
        assert_eq!(get_column(&trace.main_table, 2), a_sorted);
        assert_eq!(get_column(&trace.main_table, 3), v_sorted);
        assert_eq!(get_column(&trace.main_table, 4), m);

        // ------------------------------------------------------------------------------------
        // Define z and alpha (arbitrary constants)
        // ------------------------------------------------------------------------------------
        let z = EF::from_u64(100);
        let alpha = EF::from_u64(7);

        // ------------------------------------------------------------------------------------
        // Build auxiliary column
        // ------------------------------------------------------------------------------------
        trace.build_auxiliary_column(z, alpha);

        // ------------------------------------------------------------------------------------
        // Compute expected s values, with literal expressions in comments
        // ------------------------------------------------------------------------------------

        // Row 0
        let unsorted_0 = (z - (alpha * v[0] + a[0])).inverse();
        let sorted_0 = (z - (alpha * v_sorted[0] + a_sorted[0])).inverse();
        let s0 = sorted_0 * m[0] - unsorted_0;

        // Row 1
        // s1 = s0 + (2 / (z - (2 + alpha * 20))) - (1 / (z - (1 + alpha * 10)))
        let unsorted_1 = (z - (alpha * v[1] + a[1])).inverse();
        let sorted_1 = (z - (alpha * v_sorted[1] + a_sorted[1])).inverse();
        let s1 = s0 + sorted_1 * m[1] - unsorted_1;

        // Row 2
        // s2 = s1 + (1 / (z - (3 + alpha * 30))) - (1 / (z - (2 + alpha * 20)))
        let unsorted_2 = (z - (alpha * v[2] + a[2])).inverse();
        let sorted_2 = (z - (alpha * v_sorted[2] + a_sorted[2])).inverse();
        let s2 = s1 + sorted_2 * m[2] - unsorted_2;

        // Row 3
        // s3 = s2 + (0 / (z - (3 + alpha * 30))) - (1 / (z - (3 + alpha * 30)))
        let unsorted_3 = (z - (alpha * v[3] + a[3])).inverse();
        let sorted_3 = (z - (alpha * v_sorted[3] + a_sorted[3])).inverse();
        let s3 = s2 + sorted_3 * m[3] - unsorted_3;

        println!("tototo {:?}", vec![s0, s1, s2, s3]);

        // ------------------------------------------------------------------------------------
        // Check actual values
        // ------------------------------------------------------------------------------------
        assert_eq!(get_column(&trace.aux_table, 0), vec![s0, s1, s2, s3]);
        assert_eq!(trace.aux_table.width, 1);
        assert_eq!(trace.aux_table.height(), 4);
    }
}
