use p3_air::{AirBuilder, ExtensionBuilder, PermutationAirBuilder};
use p3_field::{ExtensionField, Field};
use p3_matrix::{dense::RowMajorMatrixView, stack::VerticalPair};

type ViewPair<'a, T> = VerticalPair<RowMajorMatrixView<'a, T>, RowMajorMatrixView<'a, T>>;

/// Builder that references slices of TraceTable for one transition step.
#[derive(Debug)]
pub struct TraceTableAirBuilder<'a, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    pub main: ViewPair<'a, F>,
    pub aux: ViewPair<'a, EF>,
    pub row_index: usize,
    pub is_first_row: F,
    pub is_last_row: F,
    pub is_transition: F,
    pub challenges: &'a [EF],
}

impl<'a, F, EF> AirBuilder for TraceTableAirBuilder<'a, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type F = F;
    type Expr = F;
    type Var = F;
    type M = ViewPair<'a, F>;

    fn main(&self) -> Self::M {
        self.main
    }

    fn is_first_row(&self) -> Self::Expr {
        self.is_first_row
    }

    fn is_last_row(&self) -> Self::Expr {
        self.is_last_row
    }

    fn is_transition_window(&self, size: usize) -> Self::Expr {
        if size == 2 {
            self.is_transition
        } else {
            panic!("only supports a window size of 2")
        }
    }

    fn assert_zero<I: Into<Self::Expr>>(&mut self, x: I) {
        let val = x.into();
        assert_eq!(val, F::ZERO, "Constraint failed at row {}", self.row_index);
    }

    fn assert_eq<I1: Into<Self::Expr>, I2: Into<Self::Expr>>(&mut self, x: I1, y: I2) {
        let x = x.into();
        let y = y.into();
        assert_eq!(
            x, y,
            "Equality failed at row {}: {:?} != {:?}",
            self.row_index, x, y
        );
    }
}

impl<F, EF> ExtensionBuilder for TraceTableAirBuilder<'_, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type EF = EF;
    type ExprEF = EF;
    type VarEF = EF;

    fn assert_zero_ext<I>(&mut self, x: I)
    where
        I: Into<Self::ExprEF>,
    {
        assert_eq!(
            x.into(),
            EF::ZERO,
            "Extension constraint failed at row {}",
            self.row_index
        );
    }
}

impl<'a, F, EF> PermutationAirBuilder for TraceTableAirBuilder<'a, F, EF>
where
    F: Field,
    EF: ExtensionField<F>,
{
    type MP = ViewPair<'a, EF>;
    type RandomVar = EF;

    fn permutation(&self) -> Self::MP {
        self.aux
    }

    fn permutation_randomness(&self) -> &[Self::RandomVar] {
        self.challenges
    }
}
