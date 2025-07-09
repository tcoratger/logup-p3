use p3_air::{Air, AirBuilder, BaseAir, ExtensionBuilder, PermutationAirBuilder};
use p3_field::{ExtensionField, Field, PrimeCharacteristicRing, PrimeField};
use p3_matrix::{Matrix, dense::RowMajorMatrixView, stack::VerticalPair};

use crate::{table::TraceTable, trace_builder::TraceTableAirBuilder};

/// Runs constraint checks for a given AIR definition and trace.
///
/// Iterates over each row, providing both the current and next row
/// (with wraparound) to the AIR logic via `TraceTableAirBuilder`.
///
/// # Arguments
/// - `air`: The AIR instance defining constraints.
/// - `trace`: The full trace table, including main and auxiliary columns.
pub(crate) fn check_constraints<F, EF, A>(air: &A, trace: &TraceTable<F, EF>)
where
    F: Field,
    EF: ExtensionField<F>,
    A: for<'a> Air<TraceTableAirBuilder<'a, F, EF>>,
{
    let n = trace.main_table.height();

    let main_view = trace.main_table.as_view();
    let aux_view = trace.aux_table.as_view();

    for i in 0..n {
        let next_i = (i + 1) % n;

        // Main trace views
        let main_local = unsafe { main_view.row_slice_unchecked(i) };
        let main_next = unsafe { main_view.row_slice_unchecked(next_i) };
        let main_pair = VerticalPair::new(
            RowMajorMatrixView::new_row(&*main_local),
            RowMajorMatrixView::new_row(&*main_next),
        );

        // Auxiliary trace views
        let aux_local = unsafe { aux_view.row_slice_unchecked(i) };
        let aux_next = unsafe { aux_view.row_slice_unchecked(next_i) };
        let aux_pair = VerticalPair::new(
            RowMajorMatrixView::new_row(&*aux_local),
            RowMajorMatrixView::new_row(&*aux_next),
        );

        let mut builder = TraceTableAirBuilder::new(main_pair, aux_pair, i, n, &[]);

        air.eval(&mut builder);
    }
}

/// LogUp AIR definition.
#[derive(Debug)]
pub struct LogUpAir<F: Field, EF: ExtensionField<F>> {
    /// Challenge z sampled in the extension field.
    pub z: EF,
    /// Challenge alpha sampled in the extension field.
    pub alpha: EF,

    pub public_inputs: LogUpPublicInputs<F>,
}

/// Public inputs struct
#[derive(Debug, Clone)]
pub struct LogUpPublicInputs<F: Field> {
    pub a0: F,
    pub v0: F,
    pub a_sorted_0: F,
    pub v_sorted_0: F,
    pub m0: F,
}

impl<F, EF> LogUpAir<F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
}

impl<F, EF> BaseAir<F> for LogUpAir<F, EF>
where
    F: Field + Sync,
    EF: ExtensionField<F> + Sync,
{
    fn width(&self) -> usize {
        5 // a, v, a_sorted, v_sorted, m
    }
}

impl<F, EF, AB> Air<AB> for LogUpAir<F, EF>
where
    F: Field + PrimeField,
    EF: ExtensionField<F>,
    AB: PermutationAirBuilder<F = F, EF = EF>,
{
    fn eval(&self, builder: &mut AB) {
        // -------------------------------------------------------------------------
        // Get references to the main and auxiliary (permutation) trace views
        // -------------------------------------------------------------------------
        let main = builder.main();
        let aux = builder.permutation();

        // -------------------------------------------------------------------------
        // Read the local and next rows for main and auxiliary traces
        // We always look at two rows to define transition constraints
        // -------------------------------------------------------------------------
        let local = main.row_slice(0).unwrap();
        let next = main.row_slice(1).unwrap();
        let aux_local = aux.row_slice(0).unwrap();
        let aux_next = aux.row_slice(1).unwrap();

        // -------------------------------------------------------------------------
        // Extract sorted columns (a', v') from the main trace at current and next rows
        // -------------------------------------------------------------------------
        let a_sorted = local[2].clone();
        let v_sorted = local[3].clone();

        let a_sorted_next = next[2].clone();
        let v_sorted_next = next[3].clone();
        let m_next = next[4].clone();

        let a_next = next[0].clone();
        let v_next = next[1].clone();

        // -------------------------------------------------------------------------
        // Extract auxiliary column s at current and next rows
        // -------------------------------------------------------------------------
        let s0 = aux_local[0];
        let s1 = aux_next[0];

        // -------------------------------------------------------------------------
        // Transition Constraint: Continuity constraint
        // Enforces that the sorted address column a' is continuous or stays the same
        // We require (a'_{i+1} - a'_i) ⋅ (a'_{i+1} - a'_i - 1) = 0
        // This enforces either equality or increment by one
        // -------------------------------------------------------------------------
        let diff_a_sorted = a_sorted_next.clone() - a_sorted;
        builder
            .when_transition()
            .assert_zero(diff_a_sorted.clone() * (diff_a_sorted.clone() - AB::Expr::ONE));

        // -------------------------------------------------------------------------
        // Transition Constraint: Single-value constraint
        // Enforces read-only memory: if address didn't change, value should not change
        // We require (v'_{i+1} - v'_i) ⋅ (a'_{i+1} - a'_i - 1) = 0
        // -------------------------------------------------------------------------
        let diff_v_sorted = v_sorted_next.clone() - v_sorted;
        builder
            .when_transition()
            .assert_zero(diff_v_sorted * (diff_a_sorted - AB::Expr::ONE));

        // -------------------------------------------------------------------------
        // Transition Constraint: Permutation constraint
        //
        // This constraint enforces the correct update of the auxiliary column s,
        // following the "LogUp" update equation.
        //
        // Original update equation (fraction form):
        //
        //     s_{i+1} = s_i + m_{i+1} / (z - (a'_{i+1} + α v'_{i+1})) - 1 / (z - (a_{i+1} + α v_{i+1}))
        //
        // We clear denominators by multiplying both sides by:
        //     (z - (a' + α v')) ⋅ (z - (a + α v))
        //
        // Resulting polynomial constraint:
        //
        //     0 = s_i ⋅ (z - (a' + α v')) ⋅ (z - (a + α v))
        //       + m ⋅ (z - (a + α v))
        //       - (z - (a' + α v'))
        //       - s_{i+1} ⋅ (z - (a' + α v')) ⋅ (z - (a + α v))
        //
        // In code, we set:
        //
        //     lhs = s_i ⋅ unsorted_term ⋅ sorted_term
        //         + m ⋅ unsorted_term
        //         - sorted_term
        //
        //     rhs = s_{i+1} ⋅ unsorted_term ⋅ sorted_term
        //
        // We enforce lhs - rhs = 0.
        // -------------------------------------------------------------------------

        let z = self.z;
        let alpha = self.alpha;

        // Convert scalar challenges to expressions
        let z_expr: AB::ExprEF = z.into();
        let alpha_expr: AB::ExprEF = alpha.into();

        // Convert field values from main and aux traces to expressions
        let a_next: AB::Expr = a_next.into();
        let v_next: AB::Expr = v_next.into();
        let a_sorted_next: AB::Expr = a_sorted_next.into();
        let v_sorted_next: AB::Expr = v_sorted_next.into();
        let m_next: AB::Expr = m_next.into();
        let s0: AB::ExprEF = s0.into(); // s_i
        let s1: AB::ExprEF = s1.into(); // s_{i+1}

        // Compute unsorted term: z - (α ⋅ v + a)
        let unsorted_term = z_expr.clone() - (alpha_expr.clone() * v_next + a_next);

        // Compute sorted term: z - (α ⋅ v' + a')
        let sorted_term = z_expr.clone() - (alpha_expr.clone() * v_sorted_next + a_sorted_next);

        // Left-hand side expression
        //
        // s_i ⋅ (z - (a' + α v')) ⋅ (z - (a + α v))
        //     + m ⋅ (z - (a + α v))
        //     - (z - (a' + α v'))
        let lhs = s0.clone() * unsorted_term.clone() * sorted_term.clone()
            + unsorted_term.clone() * m_next
            - sorted_term.clone();

        // Right-hand side expression
        //
        // s_{i+1} ⋅ (z - (a' + α v')) ⋅ (z - (a + α v))
        let rhs = s1 * unsorted_term * sorted_term;

        // Enforce: lhs - rhs = 0
        builder.when_transition().assert_zero_ext(lhs - rhs);

        // -------------------------------------------------------------------------
        // Boundary Constraints on the main trace (first row)
        //
        // These constraints enforce that the first row of the main trace exactly
        // matches the given public inputs. This ensures that the prover starts
        // the computation from a known, externally provided state.
        //
        // In the LogUp context, the main trace columns correspond to:
        //   - a:    original address column
        //   - v:    original value column
        //   - a':   sorted address column
        //   - v':   sorted value column
        //   - m:    multiplicity (used in the update)
        //
        // We enforce on the first row:
        //     a_0         = public a_0
        //     v_0         = public v_0
        //     a'_0        = public a'_0
        //     v'_0        = public v'_0
        //     m_0         = public m_0
        //
        // Each equality constraint is enforced using `when_first_row()`, so it
        // only applies on the first row (row index 0).
        // -------------------------------------------------------------------------
        let inputs = &self.public_inputs;

        builder
            .when_first_row()
            .assert_eq(local[0].clone(), inputs.a0);
        builder
            .when_first_row()
            .assert_eq(local[1].clone(), inputs.v0);
        builder
            .when_first_row()
            .assert_eq(local[2].clone(), inputs.a_sorted_0);
        builder
            .when_first_row()
            .assert_eq(local[3].clone(), inputs.v_sorted_0);
        builder
            .when_first_row()
            .assert_eq(local[4].clone(), inputs.m0);

        // -------------------------------------------------------------------------
        // Boundary Constraint: Initialize first auxiliary row (s0)
        //
        // This constraint enforces the correct initialization of the auxiliary
        // column s at the first row. In LogUp, s_0 must satisfy the following
        // polynomial identity derived from the fraction-based definition:
        //
        //     s_0 ⋅ (z - (a'_0 + α v'_0)) ⋅ (z - (a_0 + α v_0))
        //       = m_0 ⋅ (z - (a_0 + α v_0)) - (z - (a'_0 + α v'_0))
        //
        // Here:
        //   - a_0, v_0:           first row values of original columns
        //   - a'_0, v'_0:         first row values of sorted columns
        //   - m_0:                first row multiplicity
        //   - s_0:                first row auxiliary value
        //
        // The left-hand side (lhs_init) and right-hand side (rhs_init) are constructed
        // explicitly and enforced to be equal by requiring lhs_init - rhs_init = 0.
        // This ensures s_0 starts correctly and ties the auxiliary trace to the main trace.
        // -------------------------------------------------------------------------
        let v0_expr: AB::Expr = inputs.v0.into();
        let a0_expr: AB::Expr = inputs.a0.into();
        let v_sorted_0_expr: AB::Expr = inputs.v_sorted_0.into();
        let a_sorted_0_expr: AB::Expr = inputs.a_sorted_0.into();

        // Compute unsorted term: z - (α ⋅ v + a)
        let unsorted_init_term = z_expr.clone() - (alpha_expr.clone() * v0_expr + a0_expr);

        // Compute sorted term: z - (α ⋅ v' + a')
        let sorted_init_term = z_expr - (alpha_expr * v_sorted_0_expr + a_sorted_0_expr);

        let m0: AB::Expr = inputs.m0.into();

        let lhs_init = s0.clone() * unsorted_init_term.clone() * sorted_init_term.clone();
        let rhs_init = unsorted_init_term * m0 - sorted_init_term;

        builder
            .when_first_row()
            .assert_zero_ext(lhs_init - rhs_init);

        // -------------------------------------------------------------------------
        // Boundary Constraint: Final auxiliary value
        //
        // Enforce s_{n-1} = 0 on the last row.
        //
        // On the last row (i = n-1):
        //   - `s0` corresponds to s_{n-1}, the last actual auxiliary value in the trace.
        //   - `s1` corresponds to s_n, which wraps around to s_0 (first row) in cyclic accesses.
        //
        // We want to ensure that before wrapping around, the cumulative auxiliary sum
        // exactly cancels out, so we require:
        //
        //     s_{n-1} = 0.
        //
        // -------------------------------------------------------------------------
        builder.when_last_row().assert_zero_ext(s0);
    }
}

#[cfg(test)]
mod tests {
    use p3_baby_bear::BabyBear;
    use p3_field::extension::BinomialExtensionField;

    use super::*;
    use crate::table::TraceTable;

    type F = BabyBear;
    type EF = BinomialExtensionField<F, 4>;

    #[test]
    fn test_logup_air_constraints_end_to_end_documented() {
        use p3_baby_bear::BabyBear;
        use p3_field::{PrimeCharacteristicRing, extension::BinomialExtensionField};

        type F = BabyBear;
        type EF = BinomialExtensionField<F, 4>;

        // ------------------------------------------------------------------------------------
        // Define example address and value columns representing a memory trace.
        //
        // - `a` represents addresses accessed,
        // - `v` represents the corresponding values at those addresses.
        //
        // We provide small example vectors for clarity.
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

        // ------------------------------------------------------------------------------------
        // Build the initial trace table using `read_only_logup_trace`.
        //
        // This function prepares the main trace by:
        // - sorting addresses,
        // - generating sorted columns,
        // - computing multiplicities.
        //
        //  The trace is padded or adjusted if necessary to match internal requirements (e.g., power-of-two length).
        // ------------------------------------------------------------------------------------
        let mut trace = TraceTable::<F, EF>::read_only_logup_trace(&a, &v);

        // ------------------------------------------------------------------------------------
        // Choose the challenges `z` and `alpha` in the extension field EF.
        //
        // These are random challenges typically derived via a Fiat-Shamir transcript.
        // In this example, we choose fixed values to make the test deterministic.
        // ------------------------------------------------------------------------------------
        let z = EF::from_u64(123);
        let alpha = EF::from_u64(456);

        // ------------------------------------------------------------------------------------
        // Extract public input values from the first row of the main trace.
        //
        // These public inputs encode initial values that the verifier checks against
        // commitments and are part of the statement of the proof.
        // ------------------------------------------------------------------------------------
        let a0 = trace.main_table.get(0, 0).unwrap();
        let v0 = trace.main_table.get(0, 1).unwrap();
        let a_sorted_0 = trace.main_table.get(0, 2).unwrap();
        let v_sorted_0 = trace.main_table.get(0, 3).unwrap();
        let m0 = trace.main_table.get(0, 4).unwrap();

        let public_inputs = LogUpPublicInputs::<F> {
            a0,
            v0,
            a_sorted_0,
            v_sorted_0,
            m0,
        };

        // ------------------------------------------------------------------------------------
        // Build the auxiliary column of the trace using the LogUp update logic.
        //
        // This method uses the chosen challenges `z` and `alpha` to compute auxiliary
        // trace entries (the "s" column) used for permutation arguments.
        // ------------------------------------------------------------------------------------
        trace.build_auxiliary_column(z, alpha);

        // ------------------------------------------------------------------------------------
        // Create the AIR instance with all required parameters.
        //
        // The AIR struct encapsulates the random challenges and public inputs, and it
        // defines all constraint-checking logic via its `eval` method.
        // ------------------------------------------------------------------------------------
        let air = LogUpAir {
            z,
            alpha,
            public_inputs,
        };

        // ------------------------------------------------------------------------------------
        // Run the constraint checks for each row using the helper function.
        //
        // This function iterates over all rows, builds local and next row pairs,
        // and calls `air.eval()` to enforce all transition and boundary constraints.
        //
        // Internally, this checks sorted column continuity, value consistency,
        // permutation update correctness, public input matching, and proper auxiliary
        // initialization and final zero condition.
        //
        // If any constraint fails, an assertion will panic, failing the test.
        // ------------------------------------------------------------------------------------
        check_constraints(&air, &trace);

        // ------------------------------------------------------------------------------------
        // ✅ If no assertion panicked, all constraints were satisfied.
        // We print a success message for clarity in test output logs.
        // ------------------------------------------------------------------------------------
        println!("✅ All constraints checked and passed successfully!");
    }
}
