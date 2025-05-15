use lazy_static::lazy_static;

//constants that will not be changed

//unit in m/s
const C_SPEED_OF_LIGHT:f64 = 299792458.0;

//unit in m^2·kg / s, which is also j*s, which is also w*s^2
const H_PLANK_CONSTANT:f64 = 6.62607015*1e-34;

//const O_BAND_WAVELENGTH_RANGE:(LenNM,LenNM) = (1260.0,1360.0);
//const E_BAND_WAVELENGTH_RANGE:(LenNM,LenNM) = (1360.0,1460.0);

type LenNM = f64;

const S_BAND_WAVELENGTH_RANGE:(LenNM,LenNM) = (1460.0,1530.0);
const C_BAND_WAVELENGTH_RANGE:(LenNM,LenNM) = (1530.0,1565.0);
const L_BAND_WAVELENGTH_RANGE:(LenNM,LenNM) = (1565.0,1625.0);
//reference lambda, at 1550.0nm
const LAMBDA_0:LenNM = 1550.0;

pub(crate) type BandWidthTHZ = f64;
pub(crate) type DistanceKM = f64;

type FreqTHZ = f64;

//type DBPerKM = f64;

type DBM = f64;

//power, unit in Watts, must be greater than 0,
pub(crate) type PowerWatt = f64;

//free parameters, that can be tweaked
lazy_static! {
    static ref channel_spacing:BandWidthTHZ = 75.0/1000.0;
    static ref span_len:DistanceKM = 100.0;
    static ref channel_bandwidth:BandWidthTHZ = *channel_spacing;

    //dispersion at 1550nm, unit in ps/nm/km
    static ref dispersion_0:f64 = 17.0;
    //D = S*(lambda - lambda0) + D0, unit in ps/nm^2/km
    static ref dispersion_slope:f64 = 0.067;
    static ref max_power_dbm:DBM = 23.0;
}

lazy_static! {

    pub(crate) static ref center_freqs:Vec<FreqTHZ> = {
        let spacing = *channel_spacing;
        let start_freq = C_SPEED_OF_LIGHT/(L_BAND_WAVELENGTH_RANGE.1)/1e3;
        let end_freq = C_SPEED_OF_LIGHT/(S_BAND_WAVELENGTH_RANGE.0)/1e3;
        let channel_len = ((end_freq - start_freq)/spacing) as usize;
        
        let mut channels = Vec::with_capacity(channel_len);
    
        for i in 0..channel_len {
            channels.push(start_freq + spacing/2.0 + spacing*i as f64)
        }
        channels
    
    };

    pub(crate) static ref channel_power:PowerWatt = {
        const MILIWATT:PowerWatt = 0.001;
        let max_p_total = MILIWATT*10.0f64.powf(*max_power_dbm/10.0);
        max_p_total/center_freqs.len() as f64
    };

    //alpha:1/km, alpha_loss_per_band[i] gets i's band's alpha_i
    // end/start = exp(-alpha*length)
    static ref alpha_loss:Vec<f64> = {
        //α [dB/km] ≈ α2(λ − λ0)^2 + α1(λ − λ0) + α0
        //units in dB/(km·nm^2)
        let alpha_2 = 3.7685*1e-6;
        //units in dB/(km·nm)
        let alpha_1 = -7.3764*1e-5;
        //units in dB
        let alpha_0 = 0.162;
        let mut loss = Vec::with_capacity(center_freqs.len());
        for freq in center_freqs.iter() {
            let lambda:LenNM = C_SPEED_OF_LIGHT/freq/1e3;
            let deviate = lambda- LAMBDA_0;
            let alpha_db = alpha_2*(deviate.powi(2)) + alpha_1*deviate + alpha_0;

            // because alpha is used in exp(alpha*z), it has to be converted
            let alpha_loss_inner = (alpha_db/10.0)*(10.0f64.ln());
            //let alpha_loss_inner = alpha_db;
            loss.push(alpha_loss_inner)
        }
        assert_eq!(loss.len(),center_freqs.len());
        loss
    };

    static ref LeffL:Vec<DistanceKM> = {
        fn l_eff_inner(z:DistanceKM,alpha:f64) -> DistanceKM { 
            // exponnet is in (1/km)*km
            let exponent = -alpha*z;
            //1-exp(db) is unitless, and alpha is 1/KM
            return (1.0-exponent.exp())/alpha;
            //return value is KM
        }
        let mut leffl = Vec::with_capacity(center_freqs.len());
        for alpha in alpha_loss.iter() {
            leffl.push(l_eff_inner(*span_len, *alpha))
        }
        assert_eq!(leffl.len(),center_freqs.len());
        leffl
    };

    static ref sigma2_ase:Vec<PowerWatt> = {
        let mut ase = Vec::with_capacity(center_freqs.len());
        for (alpha_i,freq) in alpha_loss.iter().zip(center_freqs.iter()) {
            let g_s = ((*span_len)*alpha_i).exp();
            let lambda:LenNM = C_SPEED_OF_LIGHT/freq/1e3;
            let noise_figure_db;
            if lambda >= S_BAND_WAVELENGTH_RANGE.0 && lambda <= S_BAND_WAVELENGTH_RANGE.1 {
                noise_figure_db = 7.0;
            }else if lambda >= C_BAND_WAVELENGTH_RANGE.0 && lambda <= C_BAND_WAVELENGTH_RANGE.1 {
                noise_figure_db = 5.5;
            }else if lambda >= L_BAND_WAVELENGTH_RANGE.0 && lambda <= L_BAND_WAVELENGTH_RANGE.1 {
                noise_figure_db = 6.0;
            }else {
                panic!("wavelength {lambda} not in C+S+L band")
            };
            let n_sp = 10.0f64.powf(noise_figure_db/10.0);
            let ase_i = n_sp*H_PLANK_CONSTANT*(*channel_bandwidth*1e12)*(freq*1e12)*g_s;
            assert!(ase_i > 0.0);
            ase.push(ase_i);
        }
        assert_eq!(ase.len(),center_freqs.len());
        ase
    };
}

use std::f64::consts::PI;

lazy_static! {
    // unit in 1/(km·THZ^2), which is ps^2/km
    static ref beta_2:f64 = -1e3* (*dispersion_0)*LAMBDA_0.powi(2) / (2.0*PI*C_SPEED_OF_LIGHT);
    // unit in 1/(km·THZ^3), wich is ps^3/km
    static ref beta_3:f64 = 1e6*(LAMBDA_0/(2.0*PI*C_SPEED_OF_LIGHT)).powi(2) * ((LAMBDA_0.powi(2)* *dispersion_slope) + 2.0*LAMBDA_0* *dispersion_0);
    //unit in 1/W/KM
    static ref gamma:Vec<f64> = {
        const fn gamma_inner(freq:FreqTHZ) -> f64 {
            let lambda:LenNM = C_SPEED_OF_LIGHT/freq/1e3;
            let slope = (1.76 - 1.01)/(1350.0-1650.0);
            (lambda - 1350.0)*slope + 1.75
        }
        let mut gammas = Vec::with_capacity(center_freqs.len());
        for freq in center_freqs.iter() {
            gammas.push(gamma_inner(*freq))
        }
        assert_eq!(gammas.len(),center_freqs.len());
        gammas
    };
    static ref spm_phi:Vec<f64> = {
        let mut phi = Vec::with_capacity(center_freqs.len());
        for freq in center_freqs.iter() {
            phi.push(1.5*PI*PI*( *beta_2 + 2.0*PI*freq*(*beta_3) ))
        }
        assert_eq!(phi.len(),center_freqs.len());
        phi
    };
    //phi[i,l], called by xpm_phi[i][l]
    static ref xpm_phi:Vec<Vec<f64>> = {
        let mut phi_i_l = Vec::with_capacity(center_freqs.len());
        for (i,freq_i) in center_freqs.iter().enumerate() {
            let mut phi_l = Vec::with_capacity(center_freqs.len());
            for (l,freq_l) in center_freqs.iter().enumerate() {
                if i == l {phi_l.push(0.0)} else {
                    let coeff = (freq_l - freq_i)*( *beta_2 + PI*(*beta_3)*(freq_l + freq_i));
                    phi_l.push(2.0*PI*PI*coeff);
                }
            }
            assert_eq!(phi_l.len(),center_freqs.len());
            phi_i_l.push(phi_l)
        }
        assert_eq!(phi_i_l.len(),center_freqs.len());
        phi_i_l
    };
    static ref epsilon:Vec<f64> =  {
        let bandwidth_wdm = *channel_spacing * center_freqs.len() as f64;
        let mut v = Vec::with_capacity(center_freqs.len());
        for i in 0..center_freqs.len() {
            let l_eff = LeffL[i];
            let asinh = (PI.powi(2)/2.0)*(*beta_2)*l_eff*bandwidth_wdm.powi(2);
            let asinh = asinh.asinh();
            let epsilon_i = 1.0 + 6.0*l_eff/(*span_len)/asinh;
            let epsilon_i = (3.0/10.0)*epsilon_i.ln();
            v.push(epsilon_i)
        }
        assert_eq!(v.len(),center_freqs.len());
        v
    };
}

// units in 1/W/KM/THZ
const RAMAN_GAIN_SLOPE:f64 = 0.028;

//unit: THZ
const fn raman_shaping_profile(freq:FreqTHZ) -> FreqTHZ
{
    let f_min = C_SPEED_OF_LIGHT/(L_BAND_WAVELENGTH_RANGE.1)/1e3;
    let f_max = C_SPEED_OF_LIGHT/(S_BAND_WAVELENGTH_RANGE.0)/1e3;

    let b_t = (f_max - f_min).abs();

    let delta_f_co =  15.0;

    let f = freq;

    let p_t = 1.0; // 1.0watt/1.0watt unitless

    if f-delta_f_co < f_min && f+delta_f_co > f_max {
        //unit: Watt*THZ
        return p_t*f;
    }
    if f-delta_f_co >= f_min && f+delta_f_co <= f_max {
        return 0.0;
    }

    let coeff = p_t/b_t;
    let f_squre_over_2 = f*f*0.5;
    //unit: (Watt/THZ)*THZ^2 = Watt*THZ
    if f-delta_f_co <= f_min && f+delta_f_co <= f_max {
        let ffm = f*f_min;
        let fm2_minus_delta2_over_2 = (f_max*f_max - delta_f_co*delta_f_co)*0.5;
        return coeff*(f_squre_over_2 - ffm + fm2_minus_delta2_over_2)
    }
    if f-delta_f_co >= f_min && f+delta_f_co >= f_max {
        let ffm = f*f_max;
        let fm2_minus_delta2_over_2 = (f_min*f_min - delta_f_co*delta_f_co)*0.5;
        return coeff*(ffm - f_squre_over_2 - fm2_minus_delta2_over_2)
    }
    unreachable!()
} 

lazy_static! {
    //r(f_i)/1.0watt, a constant for each band
    //r(f_i) = r_f[i]
    static ref r_f:Vec<FreqTHZ> = {
        let mut rfi = Vec::with_capacity(center_freqs.len());
        for freq in center_freqs.iter() {
            rfi.push(raman_shaping_profile(*freq));
        }
        assert_eq!(rfi.len(),center_freqs.len());
        rfi
    };
}

#[allow(non_snake_case)]
fn T_i(i:usize,p_t:PowerWatt) -> f64 {
    let alpha_i = alpha_loss[i];
    let r_f_i = p_t*r_f[i];
    let c_r = RAMAN_GAIN_SLOPE;
    (2.0*alpha_i - c_r*r_f_i)*(2.0*alpha_i - c_r*r_f_i)
}

const PRE_EMPHASIS_K:f64 = 0.0;

fn sigma2_spm(i:usize,band_powers:&[PowerWatt],span_num:usize) -> PowerWatt {
    assert_eq!(band_powers.len(),center_freqs.len());

    let p_i = band_powers[i];

    if p_i < 1e-15 {
        return 0.0;
    }

    let alpha_i = alpha_loss[i];
    let b_i = *channel_bandwidth;
    let p_total:PowerWatt = band_powers.iter().sum();
    let phi_i = spm_phi[i];
    let t_i = T_i(i,p_total);
    let l_eff_l = LeffL[i];
    let r_f_i = p_total*r_f[i];

    let exponent = (2.0*RAMAN_GAIN_SLOPE*PRE_EMPHASIS_K*l_eff_l*r_f_i).exp();

    let asinh_1 = (phi_i*b_i.powi(2)/(PI*alpha_i)).asinh();
    let asinh_2 = (phi_i*b_i.powi(2)/(2.0*PI*alpha_i)).asinh();

    (4.0/9.0)*(p_i.powi(3)/b_i.powi(2))*(span_num as f64).powf(1.0 + epsilon[i])
    * (PI*gamma[i].powi(2)/(phi_i*3.0*alpha_i.powi(2))) * exponent
    * ( 
        ((t_i-alpha_i.powi(2))/alpha_i)*asinh_1 + 
        ((4.0*alpha_i.powi(2) - t_i)/(2.0*alpha_i))*asinh_2 
    )
}

fn sigma2_xpm(i_:usize,band_powers:&[PowerWatt],span_num:usize) -> PowerWatt {

    assert_eq!(band_powers.len(),center_freqs.len());

    #[allow(non_snake_case)]
    let T_l = T_i;

    let p_total:PowerWatt = band_powers.iter().sum();

    let p_i = band_powers[i_];

    if p_i < 1e-15 {
        return 0.0;
    }

    let b_i = *channel_bandwidth;

    let mut sum = 0.0;

    for (l,p_l) in band_powers.iter().enumerate() {
        if i_ == l {continue}
        let alpha_l = alpha_loss[l];
        let b_l = *channel_bandwidth;
        let phi_i_l = xpm_phi[i_][l];
        let t_l = T_l(l,p_total);
        let l_eff_l = LeffL[l];
        let r_f_l = p_total*r_f[l];
        let gamma_i_l = gamma[i_];

        let exponent = (2.0*RAMAN_GAIN_SLOPE*PRE_EMPHASIS_K*l_eff_l*r_f_l).exp();

        let atan_1_coeff = (t_l-alpha_l.powi(2))/alpha_l;
        let atan_1 = (phi_i_l*b_i/alpha_l).atan();
        let atan_2_coeff = (4.0*alpha_l.powi(2) - t_l)/(2.0*alpha_l);
        let atan_2 = (phi_i_l*b_i/(alpha_l*2.0)).atan();

        sum += (p_l.powi(2)/b_l)*(gamma_i_l.powi(2)/(3.0*phi_i_l*alpha_l.powi(2)))
            * exponent * (atan_1_coeff*atan_1 + atan_2_coeff * atan_2)

    }

    (32.0/27.0)*p_i*(span_num as f64)*sum
    
}

pub(crate) type NLINoise = Option<f64>;

pub(crate) fn estimate_nli(band_used:&[bool],span_num:usize) -> Vec<NLINoise> {

    assert_eq!(band_used.len(),center_freqs.len());

    let band_powers:Vec<PowerWatt> = band_used.iter().map(|item| 
        if *item {*channel_power} else {0.0}
    ).collect();
    let mut noise = Vec::with_capacity(band_powers.len());

    for (i,is_used) in band_used.iter().enumerate() {
        if *is_used {
            let spm = sigma2_spm(i, &band_powers, span_num);
            let xpm = sigma2_xpm(i, &band_powers, span_num);
            noise.push(Some(spm + xpm))
        }else{
            noise.push(None);
        }
    }
    assert_eq!(noise.len(),center_freqs.len());
    noise
}

pub fn estimate_gsnr(band_used:&[bool],span_num:usize) -> Vec<Option<f64>> {
    let nli = estimate_nli(band_used, span_num);
    let Some(srs) = srs_gain(band_used) else {
            let empty_gsnr = vec![None;center_freqs.len()];
            assert_eq!(empty_gsnr.len(),center_freqs.len());
            return empty_gsnr;
        };
    let mut gsnr = Vec::with_capacity(center_freqs.len());
    for (index,(nli,srs)) in nli.into_iter().zip(srs).enumerate() {
        let Some(nli) = nli else {
            gsnr.push(None);
            continue;
        };
        let power_ase = span_num as f64 * sigma2_ase[index]/srs;
        gsnr.push(Some(*channel_power/(power_ase + nli)))
    }
    assert_eq!(gsnr.len(),center_freqs.len());
    gsnr
}

pub(crate) type Noise = f64;
pub(crate) type DeltaNoise = f64;

use rand::Rng;

pub(crate) type UsedBandNum = usize;
pub(crate) type BandIndex = usize;
use std::collections::HashMap;

pub(crate) type NoiseSimResult = HashMap<(UsedBandNum,BandIndex),Vec<Noise>>;

pub(crate) type DeltaNoiseSimResult = HashMap<(UsedBandNum,BandIndex),Vec<DeltaNoise>>;

pub(crate) fn rand_nli_noise<R:Rng>(rng:&mut R) -> NoiseSimResult {
    let span_num = 1;
    let utilization = rng.random_range(1..999) as f64 / 1000.0;
    let used_bands:Vec<bool> = (0..center_freqs.len()).map(|_| rng.random_bool(utilization)).collect();

    let used_bands_count:usize = used_bands.iter().map(|b| *b as usize).sum();

    let noise = estimate_nli(&used_bands,span_num);

    let mut results:DeltaNoiseSimResult = HashMap::new();

    for (band_index,noise) in noise.into_iter().enumerate() {
        let Some(noise) = noise else {continue};
        if let Some(data) = results.get_mut(&(used_bands_count,band_index)) {
            data.push(noise)
        }else{
            let new_vec = vec![noise];
            results.insert((used_bands_count,band_index), new_vec);
        }
    }

    results
}

pub(crate) fn rand_gsnr<R:Rng>(rng:&mut R) -> NoiseSimResult {
    let span_num = 1;
    let utilization = rng.random_range(1..999) as f64 / 1000.0;
    let used_bands:Vec<bool> = (0..center_freqs.len()).map(|_| rng.random_bool(utilization)).collect();

    let used_bands_count:usize = used_bands.iter().map(|b| *b as usize).sum();

    let gsnr = estimate_gsnr(&used_bands,span_num);

    let mut results:DeltaNoiseSimResult = HashMap::new();

    for (band_index,gsnr) in gsnr.into_iter().enumerate() {
        let Some(gsnr) = gsnr else {continue};
        let gsnr = 10.0*gsnr.log10();
        if let Some(data) = results.get_mut(&(used_bands_count,band_index)) {
            data.push(gsnr)
        }else{
            let new_vec = vec![gsnr];
            results.insert((used_bands_count,band_index), new_vec);
        }
    }

    results
}

pub(crate) fn rand_delta_noise_ratio<R:Rng>(rng:&mut R) -> DeltaNoiseSimResult {
    let span_num = 1;
    let utilization = rng.random_range(1..999) as f64 / 1000.0;
    let (before_used_bands,free_bands,used_band_num) = loop {
        let before_used_bands:Vec<bool> = (0..center_freqs.len()).map(|_| rng.random_bool(utilization)).collect();
        let free_indexs:Vec<usize> = before_used_bands.iter()
            .enumerate().filter_map(|(index,used)| {
                if !*used {Some(index)} else {None}
            }).collect();
        if !free_indexs.is_empty() {
            let len = before_used_bands.len() - free_indexs.len();
            break (before_used_bands,free_indexs,len);
        }
    };

    let to_insert = free_bands[rng.random_range(0..free_bands.len())];

    let mut after_used_bands = before_used_bands.clone();
    after_used_bands[to_insert] = true;

    let noise_before_insert = estimate_nli(&before_used_bands,span_num);
    let noise_after_insert = estimate_nli(&after_used_bands,span_num);

    let mut results:DeltaNoiseSimResult = HashMap::new();

    for (band_index,delta_noise) in noise_before_insert
        .into_iter().zip(noise_after_insert).enumerate()
        .filter_map(|(band_index,(noise_before,noise_after))| {
            let noise_before = noise_before.as_ref()?;
            let noise_after = noise_after.as_ref()?;
            assert_ne!(noise_before,&0.0);
            assert_ne!(noise_after,&0.0);
            Some((band_index,(noise_after-noise_before)/noise_before))
        }) {
            if let Some(data) = results.get_mut(&(used_band_num,band_index)) {
                data.push(delta_noise)
            }else{
                let new_vec = vec![delta_noise];
                results.insert((used_band_num,band_index), new_vec);
            }
        };

    results
}

fn srs_gain(used_bands:&[bool]) -> Option<Vec<f64>> {
    assert_eq!(used_bands.len(),center_freqs.len());

    let p_total:PowerWatt = used_bands.iter().filter_map(|b| {
        if *b {Some(*channel_power)} else {None}
    }).sum();

    let mut base_sum = 0.0;

    for (index,used) in used_bands.iter().enumerate() {
        if !*used {continue;}
        let exp = RAMAN_GAIN_SLOPE*LeffL[index]*p_total*r_f[index];
        base_sum += (*channel_power)*(-exp).exp()
    }

    if base_sum <= 0.0 {
        return None;
    }

    let mut v = Vec::with_capacity(center_freqs.len());

    for index in 0..center_freqs.len() {

        let upper_exp = RAMAN_GAIN_SLOPE*LeffL[index]*p_total*r_f[index];

        v.push(p_total*(-upper_exp).exp()/base_sum)
    }
    assert_eq!(v.len(),center_freqs.len());
    Some(v)
}

#[cfg(test)]
mod tests {
    use super::channel_power;
    use super::{beta_2, beta_3, estimate_gsnr, srs_gain};

    use super::{center_freqs, rand_gsnr};
    use super::estimate_nli;
    use super::C_SPEED_OF_LIGHT;
    #[test]
    fn test_nli() {
        let used = vec![true;center_freqs.len()];
        println!("p_total = {}",center_freqs.len() as f64 * (*channel_power));
        for (index,nli) in estimate_nli(&used,10).into_iter().enumerate() {
            let nli = nli.unwrap();
            let nli_db = 10.0*(nli/0.001).log10();
            println!("freq {}'s nli in dbm:{nli_db}",center_freqs[index])
        }
    }
    #[test]
    fn test_beta() {
        println!("beta2:{},beta3:{}",*beta_2,*beta_3)
    }
    #[test]
    fn test_ch_power() {
        let dbm = 10.0*(*channel_power/0.001).log10();
        println!("power per ch : {}dbm", dbm)
    }
    #[test]
    fn test_srs_gain() {
        let used = vec![true;center_freqs.len()];
        println!("p_total = {}",center_freqs.len() as f64 * (*channel_power));
        use std::io::Write;
        use std::fs::OpenOptions;
        let mut f = OpenOptions::new().create(true).write(true).open("srs_gain.txt").unwrap();
        for (srs_gain,freq) in srs_gain(&used).unwrap().iter().zip(center_freqs.iter()) {
            let lambda = C_SPEED_OF_LIGHT/freq/1e3/1.0;
            f.write_fmt(format_args!("{lambda} {}\n",10.0*srs_gain.log10())).unwrap()
        }
    }
    #[test]
    fn test_gsnr() {
        let used = vec![true;center_freqs.len()];
        println!("p_total = {}",center_freqs.len() as f64 * (*channel_power));
        for (index,gsnr) in estimate_gsnr(&used, 10).into_iter().enumerate() {
            let gsnr = gsnr.unwrap();
            let freq = center_freqs[index];
            let lambda = C_SPEED_OF_LIGHT/freq/1e3;
            println!("lambda {lambda}'s gsnr is {}", 10.0*gsnr.log10())
        }
    }
    #[test]
    fn test_rand_gsnr() {
        let mut rng = rand::rng();
        rand_gsnr(&mut rng);
    }
    #[test]
    fn test_epsilon() {
        println!("{:?}",*super::epsilon)
    }
}