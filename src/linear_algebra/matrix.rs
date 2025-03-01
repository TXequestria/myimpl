use std::{borrow::Borrow, fmt::Debug, vec};
use rand::Rng;
use thiserror::Error;

#[derive(Error,Debug)]
pub enum MatrixError {
    #[error("Attempted to add or dot product two vectors that have different {len1} and {len2} length")]
    VectorLenUnmatch{len1:usize,len2:usize},
    #[error("Matrix size is:{matrix_size:?},but index at {accessed_index:?} was accessed")]
    IndexOutOfBounds{matrix_size:(usize,usize),accessed_index:(usize,usize)},
    #[error("Matrix have {row_count} rows ,but row at {accessed_row} was accessed")]
    RowOutOfBounds{row_count:usize,accessed_row:usize},
    #[error("Matrix have {col_count} cols ,but col at {accessed_col} was accessed")]
    ColOutOfBounds{col_count:usize,accessed_col:usize},
    #[error("This operation (inverse, LU) requires ({row},{col}) to be squre, which isn't")]
    NonSqureError{row:usize,col:usize},
    #[error("Matrix1 size is:{matrix_size1:?},matrix2 size is {matrix_size1:?}, there's mismatch")]
    DimensionMismatch{matrix_size1:(usize,usize),matrix_size2:(usize,usize)},
    #[error("attempted to create {row}*{col} matrix from vector/iterator with length {len}")]
    SizeMisMatch{row:usize,col:usize,len:usize},
    #[error("No pivoit found, matrix may be singular")]
    SingularError
}

type Result<T> = std::result::Result<T,MatrixError>;

trait AsVector {
    fn as_vector(&self) -> &[f64];
    fn len(&self) -> usize {
        self.as_vector().len()
    }
    fn iter(&self) -> impl Iterator<Item = f64> {
        self.as_vector().iter().map(|n| *n)
    }
    fn try_add<Rhs:AsVector + ?Sized>(&self,rhs:&Rhs) -> Result<impl AsVector> {
        if self.len() != rhs.len() {
            return Err(MatrixError::VectorLenUnmatch { len1: self.len(), len2: self.len() })
        }
        if self.len() == 0 {
            return Ok(vec![])
        }
        let v:Vec<f64> = self.iter().zip(rhs.iter()).map(|(a,b)| {a+b}).collect();
        Ok(v)
    }
    fn try_dot<Rhs:AsVector + ?Sized>(&self,rhs:&Rhs) -> Result<f64> {
        if self.len() != rhs.len() {
            return Err(MatrixError::VectorLenUnmatch { len1: self.len(), len2: self.len() })
        }
        Ok(self.iter().zip(rhs.iter()).map(|(a,b)| a*b).sum())
    }
    fn scalar_mul<Rhs:Borrow<f64>>(&self,rhs:Rhs) -> impl AsVector {
        let v:Vec<f64> = self.iter().map(|n| n*rhs.borrow()).collect();
        v
    }
}

impl<T:AsRef<[f64]>> AsVector for T {
    fn as_vector(&self) -> &[f64] {
        self.as_ref()
    }
}

trait AsMutVector : AsVector {
    fn as_mut_vector(&mut self) -> &mut [f64];
    fn iter_mut(&mut self) -> impl Iterator<Item = &mut f64> {
        self.as_mut_vector().iter_mut()
    }
    fn try_add_assign<Rhs:AsVector + ?Sized>(&mut self,rhs:&Rhs) -> Result<()> {
        if self.len() != rhs.len() {
            return Err(MatrixError::VectorLenUnmatch { len1: self.len(), len2: self.len() })
        }
        if self.len() != 0 {
            for (left,right) in self.iter_mut().zip(rhs.iter()) {
                *left += right;
            }
        }
        Ok(())
    }
    fn scalar_mul_assign<B:Borrow<f64>>(&mut self,rhs:B) {
        if self.len() == 0 {
            return;
        }
        for i in self.iter_mut() {
            *i *= rhs.borrow()
        }
    }
    fn try_swap<Rhs:AsMutVector + ?Sized>(&mut self,rhs:&mut Rhs) -> Result<()> {
        if self.len() != rhs.len() {
            return Err(MatrixError::VectorLenUnmatch { len1: self.len(), len2: self.len() })
        }
        if self.len() != 0 {
            self.as_mut_vector().swap_with_slice(rhs.as_mut_vector());
        }
        Ok(())
    }
}

impl<T:AsMut<[f64]> + AsVector> AsMutVector for T {
    fn as_mut_vector(&mut self) -> &mut [f64] {
        self.as_mut()
    }
}

// A double precision matrix, row major order
// which means rows are stored continuously
#[derive(Clone,Debug)]
pub struct Matrix {
    row_count:usize,
    col_count:usize,
    //row*col must equal elements.len()
    // otherwise invariants are broken, panics allowed
    elements:Vec<f64>
}

use std::fmt::Display;
impl Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f,"[")?;
        for (i,elem) in self.elements.iter().enumerate() {
            write!(f,"{elem}")?;
            if i+1 == self.elements.len() {
                write!(f,"]")?;
            }else if (i+1)%self.col_count == 0 {
                write!(f,",\n")?;
            }else{
                write!(f,", ")?;
            }
        }
        Ok(())
    }
}

impl Default for Matrix {
    fn default() -> Self {
        Self {
            row_count:0,
            col_count:0,
            elements:vec![]
        }
    }
}

//public implementations
impl Matrix {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn new_with_vec(v:Vec<f64>,row:usize,col:usize) -> Result<Self> {
        if row*col > v.len() {
            return Err(MatrixError::SizeMisMatch { row, col,len: v.len() })
        }
        if row*col == 0 {
            return Ok(Self::default())
        }
        let mut v = v;
        v.truncate(row*col);
        debug_assert_eq!(row*col,v.len());
        Ok(Self { row_count: row, col_count: col, elements: v })
    }
    unsafe fn get_unchecked(&self,row:usize,col:usize) -> f64 {
        unsafe {*self.elements.get_unchecked(row*self.col_count + col)}
    }
    pub fn get(&self,row:usize,col:usize) -> Result<f64> {
        debug_assert_eq!(self.row_count*self.col_count,self.elements.len());
        let out_of_bounds = MatrixError::IndexOutOfBounds { matrix_size: 
            (self.row_count,self.col_count), 
            accessed_index: (row,col) 
        };
        if row >= self.row_count || col >= self.col_count {
            return Err(out_of_bounds)
        }
        self.elements.get(row*self.col_count + col).map(|n| *n).ok_or(out_of_bounds)
    }
    pub fn get_mut(&mut self,row:usize,col:usize) -> Result<&mut f64> {
        debug_assert_eq!(self.row_count*self.col_count,self.elements.len());
        let out_of_bounds = MatrixError::IndexOutOfBounds { matrix_size: 
            (self.row_count,self.col_count), 
            accessed_index: (row,col) 
        };
        if row >= self.row_count || col >= self.col_count {
            return Err(out_of_bounds)
        }
        self.elements.get_mut(row*self.col_count + col).ok_or(out_of_bounds)
    }
    pub fn zeros(row:usize,col:usize) -> Self {
        if row*col == 0 {
            return Self::default()
        }
        Self {
            row_count:row,
            col_count:col,
            elements:vec![0.0;row*col]
        }
    }
    pub fn identity(size:usize) -> Self {
        if size == 0 {
            return Self::default()
        }
        let mut empty = vec![0.0;size*size];
        for i in 0..size {
            let index = i*size + i;
            debug_assert!(index < size*size);
            unsafe {
                *empty.get_unchecked_mut(index) = 1.0;
            }
        }
        Self {
            row_count:size,
            col_count:size,
            elements:empty
        }
    }
    // 0-1 rand
    pub fn rand<T:Rng>(row:usize,col:usize,rng:&mut T) -> Self {
        if row*col == 0 {
            return Self::default()
        }
        let mut dest = Vec::with_capacity(row*col);
        for _ in 0..row*col {
            dest.push(rng.random_range(0.0..=1.0));
        }
        debug_assert_eq!(dest.len(),row*col);
        Self {
            row_count:row,
            col_count:col,
            elements:dest
        }
    }

    fn row_exchange(&mut self,row1:usize,row2:usize,offset:usize) -> Result<()> {
        debug_assert_eq!(self.row_count*self.col_count,self.elements.len());
        let row_ofb = | row | MatrixError::RowOutOfBounds { row_count: self.row_count, accessed_row: row };
        if row1 > self.row_count {
            return Err(row_ofb(row1))
        }
        if row2 > self.row_count {
            return Err(row_ofb(row2))
        }

        if offset >= self.col_count {
            return Err(MatrixError::ColOutOfBounds { col_count: self.col_count, accessed_col: offset })
        }

        if self.row_count == 0 || self.col_count == 0 || row1 == row2{
            return Ok(());
        }

        let swap_len = self.col_count - offset;

        let row1_start = self.col_count*row1 + offset;
        let row2_start = self.col_count*row2 + offset;
        
        let row1_end = row1_start + swap_len;
        let row2_end = row2_start + swap_len;

        debug_assert!(row1_start < self.elements.len());
        debug_assert!(row2_start < self.elements.len());

        debug_assert!(row1_end <= self.elements.len());
        debug_assert!(row2_end <= self.elements.len());

        //safety: we guarded the entire matrix behind &mut self, so no race condition

        unsafe {
            let row1 =  self.elements.as_mut_ptr().add(row1_start);
            let row2 = self.elements.as_mut_ptr().add(row2_start);
            // row1 and row2 are never overlapping, or the below assertion fails
            // std::ptr::swap_nonoverlapping checks overlap
            std::ptr::swap_nonoverlapping(row1, row2, swap_len);
        };

        Ok(())
    }

    // dest_row -= coefficient*src_row
    fn row_elimination(&mut self, src_row:usize,dest_row:usize,offset:usize, coefficient:f64) -> Result<()>{
        debug_assert_eq!(self.row_count*self.col_count,self.elements.len());
        let row_ofb = | row | MatrixError::RowOutOfBounds { row_count: self.row_count, accessed_row: row };
        if src_row > self.row_count {
            return Err(row_ofb(src_row))
        }
        if dest_row > self.row_count {
            return Err(row_ofb(dest_row))
        }

        if offset >= self.col_count {
            return Err(MatrixError::ColOutOfBounds { col_count: self.col_count, accessed_col: offset })
        }

        if self.row_count == 0 || self.col_count == 0 {
            return Ok(());
        }

        let elimination_len = self.col_count - offset;

        let src_start = self.col_count*src_row + offset;
        let src_end = elimination_len + src_start;

        if src_row == dest_row {
            for i in &mut self.elements[src_start..src_end] {
                (*i)*=1.0-coefficient;
            }
            return Ok(())
        }

        let dest_start = self.col_count*dest_row + offset;
        let dest_end = elimination_len + dest_start;

        debug_assert!(src_start < self.elements.len());
        debug_assert!(dest_start < self.elements.len());
        debug_assert!(src_end <= self.elements.len());
        debug_assert!(dest_end <= self.elements.len());

        // no overlapping
        debug_assert!(dest_end <= src_start || src_end <= dest_start);

        // we have to read & write to different parts of the same &mut matrix
        // so we can't uphold the borrowing rule. go unsafe here
        let (src_row,dest_row) = unsafe {
            let src_row =  self.elements.as_ptr().add(src_start);
            let src_row = std::slice::from_raw_parts(src_row, elimination_len);
            let dest_row = self.elements.as_mut_ptr().add(dest_start);
            let dest_row = std::slice::from_raw_parts_mut(dest_row, elimination_len);
            (src_row,dest_row)
        };

        for (src_i,dest_i) in src_row.iter().zip(dest_row.iter_mut()) {
            *dest_i -= coefficient*src_i;
        }

        Ok(())
    }
    // select piviot from (pivoit_row,pivoit_column) downwards to (row_max,col)
    // column is unchanged
    fn select_piviot_row(&self,pivoit_row:usize,pivoit_col:usize) -> Result<(usize,f64)> {
        debug_assert_eq!(self.row_count*self.col_count,self.elements.len());
        if pivoit_row >= self.row_count {
            return Err(MatrixError::RowOutOfBounds { row_count: self.row_count, accessed_row: pivoit_row })
        }
        // pivoit (row,row) outside column range
        if pivoit_col >= self.col_count {
            return Err(MatrixError::ColOutOfBounds { col_count: self.col_count, accessed_col: pivoit_col })
        }

        let mut pivoit = self.get(pivoit_row, pivoit_col)?;
        let mut pivoit_row = pivoit_row;
        for next_row in pivoit_row+1..self.row_count {
            let new_pivoit = self.get(next_row, pivoit_col)?;
            if new_pivoit.abs() > pivoit.abs() {
                pivoit = new_pivoit;
                pivoit_row = next_row;
            };
            
        }
        return Ok((pivoit_row,pivoit))
    }

    // return value 1: elimination src row, elimination dest row, elimination coefficient
    // value 2: Upper triangular matrix
    // value 3: permutation vector
    fn lup_inplace(&mut self) -> Result<(Vec<(usize,usize,f64)>,Vec<usize>)> {
        debug_assert_eq!(self.row_count*self.col_count,self.elements.len());
        const EPS:f64 = 1e-12;
        if self.col_count != self.row_count {
            return Err(MatrixError::NonSqureError { row: self.row_count, col: self.col_count })
        }
        if self.col_count == 0 || self.row_count == 0 || self.elements.is_empty() {
            return Ok((vec![],vec![]))
        }
        let mut permutations:Vec<usize> = (0..self.row_count).collect();
        //let mut upper_triangle = self.clone();
        let mut elimination_vec = Vec::with_capacity(self.col_count*(self.col_count-1)/2);
        for src_row in 0..self.row_count {
            let (pivoit_row,pivoit) = self.select_piviot_row(src_row, src_row)?;
            if pivoit.abs() < EPS {return Err(MatrixError::SingularError)}
            if pivoit_row != src_row {
                permutations.swap(src_row, pivoit_row);
                self.row_exchange(src_row, pivoit_row, src_row)?;
            }
            for dest_row in src_row+1..self.row_count {
                let coef = self.get(dest_row, src_row)?/pivoit;
                if coef.abs() < EPS {continue}
                self.row_elimination(src_row, dest_row, src_row, coef)?;
                elimination_vec.push((src_row,dest_row,coef));
            }
        }
        elimination_vec.shrink_to_fit();
        Ok((elimination_vec,permutations))
    }

    pub fn lup(&self) -> Result<(Vec<(usize,usize,f64)>,Self,Vec<usize>)> {
        let mut u = self.clone();
        let (l,p) = u.lup_inplace()?;
        Ok((l,u,p))
    }

    pub fn is_empty(&self) -> bool {
        self.col_count == 0 || self.row_count == 0 || self.elements.is_empty()
    }

    pub fn dimension(&self) -> (usize,usize) {
        if self.is_empty() {return (0,0)}
        (self.row_count,self.col_count)
    }

    pub fn dot(&self,rhs:&Self) -> Result<Self> {
        if self.is_empty() || rhs.is_empty() {
            return Ok(Self::new())
        }
        if self.col_count != rhs.row_count {
            return Err(MatrixError::DimensionMismatch { 
                matrix_size1: self.dimension(), 
                matrix_size2: rhs.dimension() })
        }
        let product_row = self.row_count;
        let product_col = rhs.col_count;
        let dot_length = self.col_count;

        let mut elements = Vec::with_capacity(product_row*product_col);

        for index in 0..(product_row*product_col) {
            let i = index / product_col;
            let j = index % product_col;
            let mut vec_dot_sum = 0.0;
            for k in 0..dot_length {
                vec_dot_sum += unsafe {
                    self.get_unchecked(i,k)*rhs.get_unchecked(k,j)
                };
            }
            elements.push(vec_dot_sum);
        }

        Ok(Self {
            row_count:product_row,
            col_count:product_col,
            elements
        })
    }

    // to do
    pub fn transpose_inplace(&mut self) {
        use crate::dsa::bitset::BitSet;
        BitSet::new();
        todo!()
    }
    pub fn transpose(&self) -> Self {
        let t_row = self.col_count;
        let t_col = self.row_count;

        let mut new_vec = Vec::with_capacity(t_col*t_col);

        for index in 0..(t_col*t_row) {
            // index in new matrix is j*t_col + i;
            let j = index/t_col;
            let i = index%t_col;
            new_vec.push(
                unsafe{self.get_unchecked(i,j)}
            )
        }

        Self {
            row_count:t_row,
            col_count:t_col,
            elements:new_vec
        }

    }
    // permutation
    // start with [1,2,3,4]....
    // when row exchange, change two indexes 
    // then put [index,num] to 1
    // [2,3,1,4] means 1-> 2, 2 -> 3, so line 2 -> 1, line 3->2, 1->3
    // permutation matrix is [0,1,0,0, 
    //                        0,0,1,0,
    //                        1,0,0,0,
    //                        0,0,0,1]
    // ones at (1,2) (2,3) (3,1),(4,4)
}

#[cfg(test)]
mod tests {
    use rand::Rng;

    use super::Matrix;
    #[test]
    fn test_identity() {
        let identity_matrix = Matrix::identity(10000);
        for i in 0..10000 {
            for j in 0..10000 {
                let target_num = if i == j {1.0} else {0.0};
                assert_eq!(identity_matrix.get(i,j).unwrap(),target_num,"index at ({i},{j} failed to match)")
            }
        }
    }
    #[test]
    fn test_lu() {
        let mut rng = rand::rng();
        let size:usize = rng.random_range(20..100);
        let mut test_matrix = Matrix::rand(size, size, &mut rng);
        test_matrix.lup_inplace().unwrap();
        for i in 0..size {
            for j in 0..size {
                if i == j {assert!(test_matrix.get(i, j).unwrap().abs() > 1e-12)}
                if i > j {assert!(test_matrix.get(i, j).unwrap().abs() < 1e-12)}
            }
        }
    }
    #[test]
    fn test_transpose() {
        let mut rng = rand::rng();
        let size1:usize = rng.random_range(20..100);
        let size2:usize = rng.random_range(30..300);
        let test_matrix = Matrix::rand(size1, size2, &mut rng);
        let transpose = test_matrix.transpose();
        for i in 0..size1 {
            for j in 0..size2 {
                assert_eq!(test_matrix.get(i, j).unwrap(),transpose.get(j, i).unwrap(),"index:({i},{j})")
            }
        }
        test_matrix.dot(&transpose).unwrap();
    }
}