subroutine even_pODF_f(omega, qpoints, c, value, nPoints, nMax)
!
! Subroutine to calculate the even pODF P_n(mu)
! 
!
  implicit none
  integer, intent(in) :: nMax, nPoints
  real,    intent(in), dimension(3)         :: omega
  real,    intent(in), dimension(nPoints)   :: c
  real,    intent(in), dimension(nPoints,3) :: qpoints

  integer :: i  
  real    :: mu, evenKernel
  real, intent(out) :: value
!
  value = 0.0

  do i=1, nPoints
    mu = dot_product(omega,qpoints(i,:))
    call even_kernel_f(mu,evenKernel,nMax)
    value = value + c(i) * evenKernel  

  end do

end subroutine even_pODF_f



    

