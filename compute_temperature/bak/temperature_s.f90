subroutine get_temp(Verlet_List, N, data, T, mass)

    !!use omp_lib
    implicit none
    integer (kind = 8), intent(in) :: N
    integer(kind = 8), intent(in), dimension(0:N-1, 0:100-1) :: Verlet_List
    real(kind = 8), intent(in), dimension(0:N-1, 0:8-1) :: data
    real (kind=8), PARAMETER :: kb =  1.380649e-23
    real (kind = 8), PARAMETER :: afu = 6.022140857e23
    integer (kind = 8), PARAMETER :: dim = 3
    real(kind = 8), intent(in), dimension(0:N-1) :: mass
    real(kind = 8), intent(inout), dimension(0:N-1) :: T
    !f2py intent(in, out) :: T 
    integer (kind = 8) :: i, j, n_neigh, nei_i
    real(kind=8), dimension(0:2) :: v_neigh, v_final, v_i, v_mean
    real (kind = 8) :: ke

    do i = 0, N-1
        n_neigh = 1 ! neighbor atoms number
        v_i = data(i, 5:7)*100.0
        v_neigh = v_i
        do j = 0, size(Verlet_List, 2)-1
            nei_i = Verlet_List(i, j)
            if (nei_i > -1) then 
                n_neigh = n_neigh + 1
                v_neigh = v_neigh + data(nei_i, 5:7)*100.0
            end if 
        end do 
        v_mean = v_neigh/n_neigh
        ke = mass(i)*sum((v_i-v_mean)*(v_i-v_mean))
        do j = 0, size(Verlet_List, 2)-1
            nei_i = Verlet_List(i, j)
            if (nei_i > -1) then
                v_final = data(nei_i, 5:7)*100 - v_mean
                ke = ke + mass(nei_i)*sum(v_final*v_final)
            end if 
        end do 
        ke = ke * 0.5 /afu / 1000
        T(i) = ke * 2 / dim / n_neigh / kb

    end do 
end subroutine
