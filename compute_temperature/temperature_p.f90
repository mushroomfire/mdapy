subroutine get_temp(Verlet_List, N, vel, T, mass, n_thread, factor, max_neigh)

    use omp_lib
    implicit none
    integer (kind = 8), intent(in) :: N, n_thread, max_neigh
    integer(kind = 8), intent(in), dimension(0:N-1, 0:max_neigh-1) :: Verlet_List
    real(kind = 8), intent(in), dimension(0:N-1, 0:3-1) :: vel
    real (kind=8), PARAMETER :: kb =  1.380649e-23
    real (kind = 8), PARAMETER :: afu = 6.022140857e23
    integer (kind = 8), PARAMETER :: dim = 3
    real(kind = 8), intent(in), dimension(0:N-1) :: mass
    real(kind = 8), intent(inout), dimension(0:N-1) :: T
    !f2py intent(in, out) :: T 
    integer (kind = 8) :: i, j, n_neigh, nei_i
    real(kind=8), dimension(0:2) :: v_neigh, v_final, v_i, v_mean
    real (kind = 8) :: ke
    real(kind=8), intent(in) :: factor

    ! if (units == "real") then
    !     factor = 100000.0
    ! else if (units == "metal") then 
    !     factor = 100.0
    ! end if 
    call OMP_SET_NUM_THREADS(n_thread)
    !$omp parallel &
    !$omp shared ( vel, N, Verlet_List, mass, T) &
    !$omp private ( i, n_neigh, v_i, v_neigh, j, nei_i, v_mean, ke, v_final)
    !$omp do
    do i = 0, N-1
        n_neigh = 1 ! neighbor atoms number
        v_i = vel(i, :)*factor
        v_neigh = v_i
        do j = 0, size(Verlet_List, 2)-1
            nei_i = Verlet_List(i, j)
            if (nei_i > -1) then 
                n_neigh = n_neigh + 1
                v_neigh = v_neigh + vel(nei_i, :)*factor
            end if 
        end do 
        v_mean = v_neigh/n_neigh
        ke = mass(i)*sum((v_i-v_mean)*(v_i-v_mean))
        do j = 0, size(Verlet_List, 2)-1
            nei_i = Verlet_List(i, j)
            if (nei_i > -1) then
                v_final = vel(nei_i, :)*factor - v_mean
                ke = ke + mass(nei_i)*sum(v_final*v_final)
            end if 
        end do 
        ke = ke * 0.5 /afu / 1000
        T(i) = ke * 2 / dim / n_neigh / kb

    end do 
    !$omp end do
    !$omp end parallel
end subroutine
