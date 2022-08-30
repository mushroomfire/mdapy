subroutine pbc(Rij, Pbc_Boundary, Box_Size, Dim)
    implicit none

    integer ( kind = 8 ), intent(in) :: Dim
    real(kind=8), intent(in),dimension(0:Dim-1, 0:1) :: Box_Size
    real (kind=8), intent(in), dimension(0:Dim-1) :: Pbc_boundary
    real (kind=8), intent(inout), dimension(0:Dim-1) :: Rij
    !f2py intent(in, out) :: Rij
    real(8) :: L
    integer (kind=8) :: i

    do i = 0, Dim-1
        if (abs(Pbc_Boundary(i) - 1) < 1e-5) then
            L = Box_Size(i, 1) - Box_Size(i, 0)
            Rij(i) = Rij(i) - L*anint(Rij(i)/L)
        end if
    end do
end subroutine


subroutine compute_entropy_p(pos, vol, verlet_list, Box_Size, Pbc_boundary, cutoff, &
    sigma, nbins, use_local_density, n_thread, local_entropy, N, Dim, max_neigh)
    use omp_lib
    implicit none
    integer (kind = 8), intent(in) :: N, Dim, use_local_density, nbins, n_thread, max_neigh
    real (kind = 8), intent(in) :: vol, cutoff, sigma
    real(kind = 8), intent(in), dimension(0:N-1, 0:3-1) :: pos
    integer(kind = 8), intent(in), dimension(0:N-1, 0:max_neigh-1) :: Verlet_List
    real(kind=8), intent(in),dimension(0:Dim-1, 0:1) :: Box_Size
    real (kind=8), intent(in), dimension(0:Dim-1) :: Pbc_boundary
    real(kind=8), dimension(0:nbins-1) :: r, rsq,  prefactor, integrand, g_m
    real(kind=8), dimension(:), allocatable :: r_ij_copy
    real(kind=8), dimension(:, :), allocatable :: r_diff
    real(kind=8), intent(out), dimension(0:N-1) :: local_entropy
    !integer(kind = 8), intent(out), dimension(0:N-1) :: local_entropy
    real(kind=8), dimension(0:100-1) :: r_ij
    real(kind=8), dimension(0:3-1) :: Rij
    real(kind=8), parameter :: pi = 4.D0*DATAN(1.D0)
    real(kind=8) :: interval, global_rho, local_volume, rho, s
    integer(kind=8) :: i, j, k,  nei

    interval = cutoff/(nbins-1.0)
    r = 0.0
    local_entropy = 0.0
    do i = 0, nbins-1
        r(i) = i*interval
    end do
    global_rho = N / vol
    rsq = r**2
    prefactor = rsq * (4 * pi * global_rho * sqrt(2 * pi * sigma**2))
    prefactor(0) = prefactor(1)
    call OMP_SET_NUM_THREADS(n_thread)
    !$omp parallel &
    !$omp shared (N, Verlet_List, pos, Pbc_Boundary, Box_Size, Dim, sigma, prefactor, global_rho, local_entropy) &
    !$omp private ( i, r_ij, nei, j, Rij, r_ij_copy, r_diff, k, g_m, local_volume, rho, integrand, s)
    !$omp do
    do i = 0, N-1
        r_ij = 0.0
        do nei = 0, size(Verlet_List, 2)-1
            j = Verlet_List(i, nei)
            if ((j > -1) .and. (j /= i)) then
                Rij = pos(i, :) - pos(j, :)
                call pbc(Rij, Pbc_Boundary, Box_Size, Dim)
                r_ij(nei) = sqrt(sum(Rij**2))
            end if
        end do
        if (allocated(r_ij_copy)) deallocate(r_ij_copy)
        allocate(r_ij_copy(0:count(r_ij>0)-1))
        
        r_ij_copy = pack(r_ij, r_ij>0)
        
        if (allocated(r_diff)) deallocate(r_diff)
        allocate(r_diff(0:size(r_ij_copy)-1, 0:size(r)-1))
        r_diff = 0.0
        do k = 0, size(r_ij_copy)-1
            r_diff(k,:) = r - r_ij_copy(k)
        end do
        g_m = sum(exp(-r_diff**2/(2.0*sigma**2)), 1) / prefactor

        if (use_local_density == 1) then
            local_volume = 4./3. * pi * cutoff**3
            rho = size(r_ij_copy) / local_volume
            g_m = g_m * global_rho / rho
        else
            rho = global_rho
        end if
        
        where (g_m>=1e-10)
        integrand = (g_m * log(g_m) - g_m + 1.0)*rsq
        elsewhere
        integrand = rsq
        end where
        
        s = 0.0
        do k = 0, size(integrand)-2
            s = s + (integrand(k) + integrand(k+1)) * (r(k+1) - r(k))
        end do
        local_entropy(i) = -2.0 * pi * rho * s / 2

    end do
    !$omp end do
    !$omp end parallel
end subroutine




