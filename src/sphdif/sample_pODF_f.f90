subroutine sample_pODF_f(nsamples, nMax, qpoints, coefs, points, nPoints)
!
! Subroutine to sample the reconstructed pODF using rejection technique
! 
!
  implicit none
  integer, intent(in) :: nsamples, nMax, nPoints
  real,    intent(in), dimension(nPoints)   :: coefs
  real,    intent(in), dimension(nPoints,3) :: qpoints
 
  real, dimension(3) :: rpoint, urnumbers
  real               :: pi, mu, rsin, phi, x, y, z, sumC, kMax, k
  integer :: i, success, maxTries  
  real, intent(out), dimension(nsamples,3)  :: points
!
  pi = 3.141592653589
  maxTries = 1000*nsamples
  success  = 1

! Maximum of function
  sumC = sum(coefs)
  kMax = ( (nMax + 1.0)**2 / (4.0 * pi) ) * sumC
! 
  i = 1
  do while( ( i < maxTries) .and. (success <= nsamples) ) 

!   Get three uniform random numbers on [0,1)
    call random_number(urnumbers)
         
!   random point on the sphere
    mu   = 2.0*urnumbers(1) - 1.0
    rsin = (1.0 - mu*mu)**0.5
    phi  = 2.0*pi*urnumbers(2)

    z = mu
    y = rsin * sin(phi)
    x = rsin * cos(phi)    
    
    rpoint(1) = x
    rpoint(2) = y
    rpoint(3) = z

    call even_pODF_f(rpoint, qpoints, coefs, k, nPoints, nMax)

!   Test for acceptance
    if(kMax*urnumbers(3) < k) then
      !Accept point
      points(success,1) = x
      points(success,2) = y
      points(success,3) = z

      success = success + 1 

    end if

    i = i + 1

  end do

end subroutine sample_pODF_f



