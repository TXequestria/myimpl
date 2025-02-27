pub(crate) struct BitSet {
    size:usize,
    bytes:Vec<u8>
}

impl BitSet {
    pub(crate) fn new() -> Self {
        Self {size:0,bytes:vec![]}
    }
    pub(crate) fn with_capacity(capacity:usize) -> Self {
        if capacity == 0 {return Self::new()}
        let vec_capacity = capacity/8 + 1;
        Self {
            size:0,
            bytes:Vec::with_capacity(vec_capacity)
        }
    }
    pub(crate) fn push_bit(&mut self, bit:bool) {
        let byte_pos = self.size / 8;
        let pos_in_byte = self.size % 8;

        debug_assert!(byte_pos <= self.bytes.len());

        if let Some(byte) = self.bytes.get_mut(byte_pos) {
            let mask = 1u8 << pos_in_byte;
            if bit {
                *byte |= mask;
            }else{
                *byte &= !mask;
            }
        }else{
            self.bytes.push(if bit {127} else {0})
        }
        self.size += 1;
    }
    pub(crate) fn get_at(&self,index:usize) -> Option<bool> {
        if index >= self.size {return None}
        let byte_pos = index / 8;
        let pos_in_byte = index % 8;
        let byte = self.bytes.get(byte_pos)?;
        let mask = 1u8 << pos_in_byte;
        Some(*byte & mask > 0)
    }
    pub(crate) fn store_at(&mut self,index:usize,bit:bool) -> Option<()> {
        if index >= self.size {return None}
        let byte_pos = index / 8;
        let pos_in_byte = index % 8;
        let byte = self.bytes.get_mut(byte_pos)?;
        let mask = 1u8 << pos_in_byte;
        if bit {
            *byte |= mask;
        }else{
            *byte &= !mask;
        }
        Some(())
    }
}

#[cfg(test)]
mod tests{
    use super::BitSet;
    #[test]
    fn test_create() {
        BitSet::new();
        let mut set = BitSet::with_capacity(114514);
        for _ in 0..114514 {set.push_bit(true);}
        assert!(set.get_at(10000).unwrap());
        assert!(set.get_at(114513).unwrap());
        let bit = false;
        set.store_at(1145, bit).unwrap();
        assert_eq!(set.get_at(1145).unwrap(),bit);
    }
}