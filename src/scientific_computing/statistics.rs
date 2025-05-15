type PolyTerms = (f64,f64,f64);

fn quadratic_fit(data:&[(f64,f64)]) -> PolyTerms {
    if data.len() < 3 {
        panic!("at least 3 points");
    } 

    let n = data.len() as f64;

    let mut sum_x = 0.0;
    let mut sum_x2 = 0.0;
    let mut sum_x3 = 0.0;
    let mut sum_x4 = 0.0;

    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_x2y = 0.0;

    for &(xi,yi) in data {
        sum_x += xi;
        sum_x2 += xi.powi(2);
        sum_x3 += xi.powi(3);
        sum_x4 += xi.powi(4);

        sum_y += yi;
        sum_xy += xi*yi;
        sum_x2y += xi.powi(2)*yi;
    }

    let denominator = (n*sum_x2*sum_x4) - (n*sum_x3.powi(2))
        - (sum_x.powi(2)*sum_x4) + (2.0*sum_x*sum_x2*sum_x3) 
        - sum_x2.powi(3);

    if denominator.abs() < 1e-12 {
        panic!("denominator is 0");
    }

    let coeff_a = (sum_y*(sum_x*sum_x3 - sum_x2.powi(2))
        + sum_xy*(sum_x*sum_x2 - n*sum_x3)
        + sum_x2y*(n*sum_x2 - sum_x.powi(2)))/denominator;

    let coeff_b = (sum_y*(sum_x2*sum_x3 - sum_x*sum_x4)
        + sum_xy*(n*sum_x4 - sum_x2.powi(2))
        + sum_x2y*(sum_x*sum_x2 - n*sum_x3))/denominator;

    let coeff_c = (sum_y*(sum_x2*sum_x4 - sum_x3.powi(2))
        + sum_xy*(sum_x2*sum_x3 - sum_x*sum_x4)
        + sum_x2y*(sum_x*sum_x3 - sum_x2.powi(2)))/denominator;
    
    (coeff_a,coeff_b,coeff_c)
}

type FitError = (f64,f64);
type FitResult = (PolyTerms,FitError);

fn qfit_and_examine(data:Vec<(f64,f64)>) -> FitResult {

    assert!(data.len() > 3);

    let (a,b,c) = quadratic_fit(&data);

    let n = data.len() as f64;

    let y_mean = data.iter().map(|(_,y)| y).sum::<f64>()/n;

    let mut rss = 0.0;
    let mut tss = 0.0;
    for (x,y) in data {
        rss += (y - (a*x.powi(2) + b*x + c)).powi(2);
        tss += (y - y_mean).powi(2);
    }
    let rsme = (rss/(n - 3.0) as f64).sqrt();
    let r_squre = 1.0 - rss/tss;
    ((a,b,c),(rsme,r_squre))
}

#[cfg(test)]
mod test {
    #[test]
    fn test_fit() {
        let mut rng = rand::rng();
        use rand::Rng;
        for _ in 0..1 {
            let a = rng.random_range(-1000.0..1000.0);
            let b = rng.random_range(-1000.0..1000.0);
            let c = rng.random_range(-1000.0..1000.0);

            let len = rng.random_range(100..10000);

            let data:Vec<(f64,f64)> = (0..len).map(|_| {
                let x = rng.random_range(-10000.0..10000.0);
                let y = a*x*x + b*x + c;
                (x,y)
            }).collect();

            let (_,(rsme,r2)) = super::qfit_and_examine(data);

            println!("rsme:{rsme},r2:{r2}")

        }
        // pass!
    }
}